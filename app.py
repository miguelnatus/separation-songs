import os
import shutil
import tempfile
import subprocess
from pathlib import Path
import gradio as gr

# Modelos populares do Demucs (pode acrescentar outros)
DEMUCS_MODELS = [
    "htdemucs",        # bom equilíbrio
    "htdemucs_ft",     # finetuned
    "mdx",             # linha MDX (variações)
    "mdx_q",
    "mdx_extra",
    "mdx_extra_q",
]

def ensure_ffmpeg_in_path():
    """Tenta localizar o ffmpeg se não estiver no PATH (Windows)."""
    # Muitos installs do ffmpeg via winget/choco já colocam no PATH.
    # Se quiser, adicione aqui paths manuais.
    return shutil.which("ffmpeg") is not None

def separate_audio(audio_file, model_name, use_gpu, progress=gr.Progress(track_tqdm=True)):
    """
    Executa Demucs via CLI para separar o áudio e retorna um ZIP com os stems.
    """
    progress(0, desc="Preparando...")
    if audio_file is None:
        raise gr.Error("Envie um arquivo de áudio.")

    if not ensure_ffmpeg_in_path():
        raise gr.Error("FFmpeg não encontrado no PATH. Instale com 'winget install Gyan.FFmpeg' e reinicie o terminal.")

    # Cria diretórios temporários
    workdir = Path(tempfile.mkdtemp(prefix="demucs_work_"))
    uploads = workdir / "uploads"
    outputs = workdir / "separated"
    uploads.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)

    # Salva o arquivo de entrada
    in_path = uploads / Path(audio_file).name
    shutil.copyfile(audio_file, in_path)

    # Monta comando Demucs
    # Obs: demucs cria estrutura separated/<model>/<basename>/{drums,bass,vocals,other,...}
    cmd = ["demucs", "-n", model_name, "-o", str(outputs)]
    if not use_gpu:
        cmd += ["--cpu"]

    cmd.append(str(in_path))

    progress(0.15, desc="Rodando Demucs...")
    try:
        # Executa Demucs
        completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        # Mostra parte do stderr para debug
        err = e.stderr[-1000:] if e.stderr else str(e)
        raise gr.Error(f"Falha ao executar Demucs.\n\n{err}")

    progress(0.75, desc="Compactando resultados...")

    # Encontra a pasta gerada (model/basename)
    # outputs / <model> / <basename>
    model_dir = outputs / model_name
    if not model_dir.exists():
        # Alguns modelos podem salvar com outro nome; fallback: pega primeira pasta
        subdirs = [p for p in outputs.iterdir() if p.is_dir()]
        if subdirs:
            model_dir = subdirs[0]
        else:
            raise gr.Error("Não encontrei a saída do Demucs. Verifique logs.")

    # A subpasta do arquivo
    children = [p for p in model_dir.iterdir() if p.is_dir()]
    if not children:
        raise gr.Error("Não encontrei as pastas de stems. Verifique o arquivo de entrada e o modelo.")
    stems_dir = children[0]

    # Cria zip dos stems
    zip_base = workdir / "stems"
    zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=str(stems_dir))

    progress(1.0, desc="Pronto!")

    # Retorna o ZIP e também mostra a pasta (opcional)
    return zip_path

with gr.Blocks(title="Separador de Stems (Demucs)") as demo:
    gr.Markdown("# 🎛️ Separação de Stems com Demucs (PyTorch + Gradio)")
    gr.Markdown("Envie um arquivo de áudio e selecione o modelo. Receba um ZIP com os stems (vocals, drums, bass, other, etc.).")

    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Áudio de entrada (mp3/wav/flac...)")
    with gr.Row():
        model_dropdown = gr.Dropdown(choices=DEMUCS_MODELS, value="htdemucs", label="Modelo")
        use_gpu = gr.Checkbox(value=False, label="Usar GPU (CUDA)", info="Marque se tiver NVIDIA + CUDA configurados")

    run_btn = gr.Button("Separar")
    zip_output = gr.File(label="Baixar ZIP com Stems")

    run_btn.click(
        separate_audio,
        inputs=[audio_input, model_dropdown, use_gpu],
        outputs=[zip_output]
    )

if __name__ == "__main__":
    # server_name="0.0.0.0" se quiser acessar de outro dispositivo na rede
    demo.launch(share=False)
