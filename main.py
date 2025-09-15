# main.py
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional, Tuple, Dict

import gradio as gr
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

# ============================
# Static & App bootstrap
# ============================

BASE_DIR = Path(__file__).parent.resolve()
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)  # garante a pasta

app = FastAPI(title="Separador de Stems – DevSounds")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ============================
# Configurações / Utilidades
# ============================

DEMUCS_MODELS = [
    "htdemucs",
    "htdemucs_ft",
    "mdx",
    "mdx_extra",
]
STEM_KEYS = ["vocals", "drums", "bass", "other"]

def ensure_ffmpeg_in_path() -> bool:
    return shutil.which("ffmpeg") is not None

def slugify_basename(name: str, default: str = "input") -> str:
    base = Path(name).stem
    base = base.lower()
    base = re.sub(r"[^\w\s-]", "", base)
    base = re.sub(r"[-\s]+", "-", base).strip("-_")
    return base or default

def find_stems(stems_dir: Path) -> Dict[str, Optional[str]]:
    paths: Dict[str, Optional[str]] = {k: None for k in STEM_KEYS}
    wavs = list(stems_dir.glob("*.wav"))

    for w in wavs:
        name = w.stem.lower()
        for key in STEM_KEYS:
            if key in name:
                paths[key] = str(w)

    if not any(paths.values()) and wavs:
        wavs_sorted = sorted(wavs)
        for i, key in enumerate(STEM_KEYS):
            if i < len(wavs_sorted):
                paths[key] = str(wavs_sorted[i])

    return paths

def run_demucs(input_path: Path, model_name: str, use_gpu: bool, out_root: Path) -> Path:
    if model_name not in DEMUCS_MODELS:
        raise ValueError(f"Modelo inválido: {model_name}. Use um destes: {', '.join(DEMUCS_MODELS)}")

    device = "cuda" if use_gpu else "cpu"
    cmd = ["demucs", "-n", model_name, "-o", str(out_root), "-d", device, str(input_path)]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        hint = []
        if "argument" in stderr and "--cpu" in stderr:
            hint.append("A flag --cpu não existe nessa versão. Use -d cpu/cuda.")
        if "not available" in stderr.lower() and device == "cuda":
            hint.append("CUDA não disponível: instale PyTorch com CUDA ou use -d cpu.")
        msg = "Falha ao executar Demucs.\n" + (stderr[-2000:] if stderr else str(e))
        if hint:
            msg += "\n\nDicas:\n- " + "\n- ".join(hint)
        raise RuntimeError(msg)

    model_dir = out_root / model_name
    if not model_dir.exists():
        subdirs = [p for p in out_root.iterdir() if p.is_dir()]
        if not subdirs:
            raise RuntimeError("Saída do Demucs não encontrada. Verifique logs.")
        model_dir = subdirs[0]

    children = [p for p in model_dir.iterdir() if p.is_dir()]
    for child in children:
        if list(child.glob("*.wav")):
            return child

    if list(model_dir.glob("*.wav")):
        return model_dir

    for p in model_dir.rglob("*.wav"):
        return p.parent

    raise RuntimeError("Pastas/arquivos de stems não encontrados na saída do Demucs.")

def zip_dir(src_dir: Path, zip_base: Path) -> Path:
    zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=str(src_dir))
    return Path(zip_path)

def separate_core(file_path: Path, model_name: str = "htdemucs", use_gpu: bool = False) -> Tuple[Path, Path, Path]:
    if not ensure_ffmpeg_in_path():
        raise RuntimeError(
            "FFmpeg não encontrado no PATH. "
            "Instale com 'winget install Gyan.FFmpeg' (ou 'choco install ffmpeg') e reabra o terminal."
        )

    workdir = Path(tempfile.mkdtemp(prefix="demucs_work_"))
    uploads = workdir / "uploads"
    outputs = workdir / "separated"
    uploads.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)

    safe_base = slugify_basename(file_path.name, default="input")
    in_path = uploads / f"{safe_base}.wav"
    shutil.copyfile(file_path, in_path)

    stems_dir = run_demucs(in_path, model_name, use_gpu, outputs)
    zip_path = zip_dir(stems_dir, workdir / safe_base)
    return Path(zip_path), workdir, stems_dir

def safe_rmtree(path: Path):
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass

# ============================
# FastAPI (API)
# ============================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ajuste conforme necessário
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/separate")
async def api_separate(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Arquivo de áudio"),
    model: str = Form("htdemucs"),
    use_gpu: bool = Form(False),
):
    suffix = Path(file.filename).suffix or ".wav"
    tmp_in_dir = Path(tempfile.mkdtemp(prefix="api_in_"))
    tmp_in = tmp_in_dir / f"upload{suffix}"

    workdir: Optional[Path] = None
    try:
        content = await file.read()
        tmp_in.write_bytes(content)

        zip_path, workdir, _stems_dir = separate_core(tmp_in, model, use_gpu)

        def _cleanup():
            safe_rmtree(workdir)
            safe_rmtree(tmp_in_dir)

        background_tasks.add_task(_cleanup)

        filename = f"{slugify_basename(file.filename)}_{model}_stems.zip"
        return FileResponse(
            path=str(zip_path),
            media_type="application/zip",
            filename=filename
        )
    except ValueError as ve:
        safe_rmtree(tmp_in_dir)
        if workdir:
            safe_rmtree(workdir)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        safe_rmtree(tmp_in_dir)
        if workdir:
            safe_rmtree(workdir)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
def api_models():
    return {"models": DEMUCS_MODELS}

# ============================
# Gradio (UI) montado no FastAPI
# ============================

# Tema enxuto: apenas chaves suportadas amplamente na API da sua versão.
THEME = gr.themes.Soft(
    primary_hue="violet",
    neutral_hue="slate",
).set(
    body_background_fill="#0B1020",
    body_text_color="#EAF0FF",
    body_text_color_subdued="#A5B0D0",
    block_background_fill="#121931",
    block_border_width="1px",
    block_border_color="#1E2A55",
    button_primary_background_fill="#7C5CFF",
    button_primary_text_color="#0B1020",
    button_primary_background_fill_hover="#8A6BFF",
)

# Raios de borda, espaçamentos e demais tokens ficam 100% no CSS,
# evitando incompatibilidades entre versões do Gradio.
CUSTOM_CSS = """
:root{
  --ds-bg:#0B1020; --ds-surface:#121931; --ds-surface-2:#172141;
  --ds-text:#EAF0FF; --ds-text-2:#A5B0D0;
  --ds-primary:#7C5CFF; --ds-accent:#11D8C3;

  --ds-radius: 16px;
  --ds-radius-sm: 12px;
  --ds-gap: 16px;
  --ds-gap-lg: 24px;
}

.gradio-container{ max-width:1080px; margin:auto; padding:16px; }
.topbar{ display:flex; gap:12px; align-items:center; padding:12px 0; }
.topbar h1{ font-size: clamp(20px, 2.5vw, 28px); margin:0; color:var(--ds-text);}

.card{
  background: linear-gradient(180deg, var(--ds-surface), var(--ds-surface-2));
  border:1px solid rgba(255,255,255,.06);
  border-radius: var(--ds-radius);
  padding:20px;
}

.upload-box{
  border:1.5px dashed rgba(255,255,255,.18);
  padding:24px;
  border-radius: var(--ds-radius);
}

.cta{ display:flex; flex-wrap:wrap; gap: var(--ds-gap); align-items:flex-start; }

button.primary{
  background:var(--ds-primary) !important;
  color:#0B1020 !important;
  border-radius: var(--ds-radius-sm) !important;
}

small.helper{ color:var(--ds-text-2); display:block; margin-top:8px; }
.tab-help{ color:var(--ds-text-2); margin-top:6px;}
.progress-aria{ position:absolute; left:-9999px; top:auto; width:1px; height:1px; overflow:hidden; }

.audio-container, .tabs, .tabitem, .block, .form, .group{
  border-radius: var(--ds-radius-sm);
}

@media (max-width: 720px){
  .cta{ flex-direction:column; }
}
"""


def gradio_workflow(audio_file: Optional[str], model_name: str, use_gpu: bool, progress=gr.Progress(track_tqdm=True)):
    if not audio_file:
        raise gr.Error("Envie um arquivo de áudio.")
    progress(0.1, desc="Preparando…")
    try:
        zip_path, workdir, stems_dir = separate_core(Path(audio_file), model_name, use_gpu)
        stems = find_stems(stems_dir)
    except Exception as e:
        raise gr.Error(str(e))
    progress(1.0, desc="Pronto!")

    def _cleanup_later():
        safe_rmtree(workdir)
    threading.Timer(120.0, _cleanup_later).start()

    # ordem fixa para tabs
    return stems["vocals"], stems["drums"], stems["bass"], stems["other"], str(zip_path)

with gr.Blocks(theme=THEME, css=CUSTOM_CSS, title="Separar voz e instrumentos (stems) – DevSounds") as demo:
    # Topbar
    gr.HTML(
        """
        <div class="topbar" role="banner">
          <img src="/static/logo.png" alt="DevSounds" style="height:32px; width:auto; border-radius:8px;">
          <h1>Separador de Stems</h1>
          <div style="margin-left:auto"><a href="#como-funciona" style="color:#A5B0D0">Como funciona?</a></div>
        </div>
        """
    )

    # Hero / CTA
    with gr.Group(elem_classes="card"):
        gr.Markdown("## Extraia vozes e instrumentos\nArraste seu arquivo ou clique para enviar. Depois, escolha o modelo e clique em **Separar**.")
        with gr.Row(elem_classes="cta"):
            audio_input = gr.Audio(type="filepath", label="Áudio (mp3/wav/flac)", elem_classes="upload-box")
            with gr.Column():
                model_dropdown = gr.Dropdown(choices=DEMUCS_MODELS, value="htdemucs", label="Modelo")
                use_gpu = gr.Checkbox(value=False, label="Usar GPU (CUDA)", info="Marque se tiver NVIDIA + CUDA configurado")
                run_btn = gr.Button("Separar", elem_classes="primary")
                gr.HTML('<small class="helper">Formatos: mp3, wav, flac. Arquivos temporários são removidos automaticamente após o processamento.</small>')

    # Resultados (Tabs)
    gr.Markdown("### 🎧 Stems")
    with gr.Tabs(elem_classes="tabs"):
        with gr.Tab("Voz"):
            out_vocals = gr.Audio(label="Voz", interactive=False, show_download_button=True)
            gr.HTML('<div class="tab-help">Faixa de voz isolada (canto, backing vocals).</div>')
        with gr.Tab("Bateria"):
            out_drums = gr.Audio(label="Bateria", interactive=False, show_download_button=True)
        with gr.Tab("Baixo"):
            out_bass = gr.Audio(label="Baixo", interactive=False, show_download_button=True)
        with gr.Tab("Outros"):
            out_other = gr.Audio(label="Outros", interactive=False, show_download_button=True)
        with gr.Tab("ZIP"):
            zip_output = gr.File(label="Baixar ZIP com todos os stems")

    # Adsense (abaixo da dobra para melhor UX/SEO)
    gr.HTML(
        """
        <div style="margin: 24px 0; text-align:center;">
          <ins class="adsbygoogle"
               style="display:block"
               data-ad-client="ca-pub-5326521012969106"
               data-ad-slot="1234567890"
               data-ad-format="auto"
               data-full-width-responsive="true"></ins>
          <script>(adsbygoogle = window.adsbygoogle || []).push({});</script>
        </div>
        """
    )

    # Sessão educativa
    gr.Markdown("### 📚 Guia rápido")
    with gr.Row():
        gr.Markdown("#### Como separar\n1. Envie o áudio. 2. Escolha o modelo. 3. Clique em **Separar**.\n\n> Dica: use áudio estéreo de boa qualidade para melhores resultados.")
        gr.Markdown("#### Qual modelo escolher?\n- **htdemucs**: equilíbrio geral.\n- **mdx_extra**: ênfase em vocais (pode ser mais lento).\n- **htdemucs_ft / mdx**: alternativas quando os outros falharem em seu material.")
        gr.Markdown("#### Privacidade\nOs arquivos são processados e **apagados automaticamente** após um curto período. Baixe o ZIP para manter uma cópia local.")

    # Âncora "Como funciona?"
    gr.Markdown('<div id="como-funciona"></div>')
    with gr.Accordion("Mais detalhes técnicos (opcional)", open=False):
        gr.Markdown(
            "- O processamento usa **Demucs**. Se **CUDA** estiver disponível, selecione **Usar GPU**.\n"
            "- **FFmpeg** é necessário para lidar com diferentes formatos.\n"
            "- Em caso de erro, verifique a mensagem e as dicas exibidas."
        )

    # Região de progresso com aria-live para a11y (screen readers)
    gr.HTML('<div class="progress-aria" aria-live="polite" id="live-status">Processo iniciado</div>')

    # Ação
    run_btn.click(
        gradio_workflow,
        inputs=[audio_input, model_dropdown, use_gpu],
        outputs=[out_vocals, out_drums, out_bass, out_other, zip_output]
    )

# Monta o Gradio dentro do FastAPI na RAIZ
gr.mount_gradio_app(app, demo, path="/")

# Rota simples para teste de saúde/SEO (pode expandir OG/JSON-LD se quiser)
@app.get("/", response_class=HTMLResponse)
def root():
    # Gradio monta na raiz acima, mas manter esta rota ajuda em setups específicos
    # e dá um fallback SEO minimalista (caso queira customizar o head em reverse proxy).
    return HTMLResponse("<!doctype html><html><head><link rel='icon' href='/static/favicon.ico' type='image/x-icon'><meta charset='utf-8'><title>DevSounds Stems</title></head><body>App carregando…</body></html>")

# Execução:
# uvicorn main:app --reload
