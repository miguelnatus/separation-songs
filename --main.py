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
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

# ============================
# Static & App bootstrap
# ============================

BASE_DIR = Path(__file__).parent.resolve()
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)  # garante a pasta
# CUSTOM_CSS = """
#     gradio-app {
#        background: #0a0a0a !important;
#     }
#     .audio-container.svelte-1ud6e7m {
#        border: 1px solid rgba(255, 255, 255, 0.1); 
#        background: #1a1a1a;
#     }
# """

app = FastAPI(title="Separador de faixas - DEVSOUNDS -")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ============================
# Configurações / Utilidades
# ============================

# Modelos suportados pela sua versão do Demucs (confira com: demucs -h)
DEMUCS_MODELS = [
    "htdemucs",
    "htdemucs_ft",
    "mdx",
    "mdx_extra",
]

STEM_KEYS = ["vocals", "drums", "bass", "other"]

def ensure_ffmpeg_in_path() -> bool:
    """Verifica se ffmpeg está no PATH (necessário para vários formatos)."""
    return shutil.which("ffmpeg") is not None

def slugify_basename(name: str, default: str = "input") -> str:
    """Gera um nome ASCII seguro para arquivos/pastas."""
    base = Path(name).stem
    base = base.lower()
    base = re.sub(r"[^\w\s-]", "", base)
    base = re.sub(r"[-\s]+", "-", base).strip("-_")
    return base or default

def find_stems(stems_dir: Path) -> Dict[str, Optional[str]]:
    """
    Procura arquivos .wav dos stems no diretório e retorna paths por chave.
    Retorna None para stems ausentes.
    """
    paths: Dict[str, Optional[str]] = {k: None for k in STEM_KEYS}
    wavs = list(stems_dir.glob("*.wav"))

    # tenta mapear pelo nome do arquivo
    for w in wavs:
        name = w.stem.lower()
        for key in STEM_KEYS:
            if key in name:
                paths[key] = str(w)

    # fallback: se não achou nada por nome, só preenche por ordem
    if not any(paths.values()) and wavs:
        wavs_sorted = sorted(wavs)
        for i, key in enumerate(STEM_KEYS):
            if i < len(wavs_sorted):
                paths[key] = str(wavs_sorted[i])

    return paths

def run_demucs(input_path: Path, model_name: str, use_gpu: bool, out_root: Path) -> Path:
    """
    Executa Demucs via CLI, salva em out_root e retorna o diretório onde estão os stems.
    Estrutura típica: out_root/<model>/<track>/{vocals,drums,bass,other}.wav
    """
    if model_name not in DEMUCS_MODELS:
        raise ValueError(
            f"Modelo inválido: {model_name}. Use um destes: {', '.join(DEMUCS_MODELS)}"
        )

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

    # Descobrir a pasta de saída
    model_dir = out_root / model_name
    if not model_dir.exists():
        # Fallback: se o nome do modelo vier diferente
        subdirs = [p for p in out_root.iterdir() if p.is_dir()]
        if not subdirs:
            raise RuntimeError("Saída do Demucs não encontrada. Verifique logs.")
        model_dir = subdirs[0]

    # Caso padrão: existem subpastas por track
    children = [p for p in model_dir.iterdir() if p.is_dir()]
    for child in children:
        if list(child.glob("*.wav")):
            return child

    # Fallback: se, por algum motivo, os WAVs estiverem direto no model_dir
    if list(model_dir.glob("*.wav")):
        return model_dir

    # Último recurso: procura recursiva
    for p in model_dir.rglob("*.wav"):
        return p.parent

    raise RuntimeError("Pastas/arquivos de stems não encontrados na saída do Demucs.")

def zip_dir(src_dir: Path, zip_base: Path) -> Path:
    """Compacta src_dir em zip_base.zip e retorna o caminho do zip."""
    zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=str(src_dir))
    return Path(zip_path)

def separate_core(file_path: Path, model_name: str = "htdemucs", use_gpu: bool = False) -> Tuple[Path, Path, Path]:
    """
    Pipeline completo: cria diretório temporário, copia input (ASCII seguro),
    roda demucs e devolve (zip_path, workdir, stems_dir) para permitir limpeza posterior.
    """
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
    # força nome ASCII e extensão .wav para evitar bugs de path no Windows
    in_path = uploads / f"{safe_base}.wav"
    shutil.copyfile(file_path, in_path)

    stems_dir = run_demucs(in_path, model_name, use_gpu, outputs)
    zip_path = zip_dir(stems_dir, workdir / safe_base)  # ex.: .../demucs_work_xxx/<safe_base>.zip
    return Path(zip_path), workdir, stems_dir

def safe_rmtree(path: Path):
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass

# ============================
# FastAPI (API)
# ============================

# CORS básico – útil se quiser chamar de outro front
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ajuste conforme necessário
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
    """Recebe arquivo via multipart/form-data e retorna um .zip com os stems. Limpa tudo após enviar."""
    suffix = Path(file.filename).suffix or ".wav"
    tmp_in_dir = Path(tempfile.mkdtemp(prefix="api_in_"))
    tmp_in = tmp_in_dir / f"upload{suffix}"

    workdir: Optional[Path] = None
    try:
        content = await file.read()
        tmp_in.write_bytes(content)

        zip_path, workdir, _stems_dir = separate_core(tmp_in, model, use_gpu)

        # Agenda limpeza completa do workdir e do upload temporário
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
        # limpeza em erro
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

def gradio_workflow(audio_file: Optional[str], model_name: str, use_gpu: bool, progress=gr.Progress(track_tqdm=True)):
    if not audio_file:
        raise gr.Error("Envie um arquivo de áudio.")
    progress(0.05, desc="Preparando...")
    try:
        zip_path, workdir, stems_dir = separate_core(Path(audio_file), model_name, use_gpu)
        stems = find_stems(stems_dir)
    except Exception as e:
        raise gr.Error(str(e))
    progress(1.0, desc="Pronto!")

    # agenda limpeza do workdir (e tudo dentro) após 2 minutos
    def _cleanup_later():
        safe_rmtree(workdir)

    threading.Timer(120.0, _cleanup_later).start()

    # Retorna na ordem fixa: vocals, drums, bass, other, zip
    return stems["vocals"], stems["drums"], stems["bass"], stems["other"], str(zip_path)

# with gr.Blocks(title="Separador de faixas - DEVSOUNDS -", css=CUSTOM_CSS) as demo:
with gr.Blocks(title="Separador de faixas - DEVSOUNDS -") as demo:
    # Header com logo (lido de /static/logo.png)

    gr.HTML(
        """
        <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-5326521012969106"
                crossorigin="anonymous"></script>
        """,
        elem_id="adsense-head"
    )

    gr.HTML(
        """
        <div class="topbar" style="display:flex;align-items:center;gap:12px; padding:12px 0;">
          <img src="/static/logo.png" alt="DevSounds Logo" style="height:40px; width:auto; border-radius:8px;">
          <h1 style="margin:0; font-size:1.5rem;">Separador de Faixas - DEVSOUNDS</h1>
        </div>
        """
    )

    gr.Markdown(
        "Envie um áudio, escolha o modelo e **ouça os stems** abaixo. "
        "Também disponibilizamos um **ZIP** para download.\n\n"
        ""
    )

    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Áudio de entrada (mp3/wav/flac...)")
    with gr.Row():
        model_dropdown = gr.Dropdown(choices=DEMUCS_MODELS, value="htdemucs", label="Modelo")
        use_gpu = gr.Checkbox(value=False, label="Usar GPU (CUDA)", info="Marque se tiver NVIDIA + CUDA configurado")

    run_btn = gr.Button("Separar")

    gr.HTML(
        """
        <div style="margin: 20px 0; text-align:center;">
        <ins class="adsbygoogle"
            style="display:block"
            data-ad-client="ca-pub-5326521012969106"
            data-ad-slot="1234567890"
            data-ad-format="auto"
            data-full-width-responsive="true"></ins>
        <script>
            (adsbygoogle = window.adsbygoogle || []).push({});
        </script>
        </div>
        """
    )

    gr.Markdown("### 🎧 Stems")
    with gr.Row():
        out_vocals = gr.Audio(label="Vocals", interactive=False, show_download_button=True)
        out_drums  = gr.Audio(label="Drums",  interactive=False, show_download_button=True)
    with gr.Row():
        out_bass   = gr.Audio(label="Bass",   interactive=False, show_download_button=True)
        out_other  = gr.Audio(label="Other",  interactive=False, show_download_button=True)

    zip_output = gr.File(label="Baixar ZIP com Stems")

    run_btn.click(
        gradio_workflow,
        inputs=[audio_input, model_dropdown, use_gpu],
        outputs=[out_vocals, out_drums, out_bass, out_other, zip_output]
    )

# Monta o Gradio dentro do FastAPI na RAIZ
gr.mount_gradio_app(app, demo, path="/")

# Execução:
# uvicorn main:app --reload
