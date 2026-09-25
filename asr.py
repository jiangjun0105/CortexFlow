import numpy as np

MODEL = "mlx-community/whisper-large-v3-turbo"


def transcribe(pcm: np.ndarray) -> str:
    """float32 mono 16 kHz -> stripped text. "" on empty input or any error."""
    if pcm is None or len(pcm) == 0:
        return ""
    try:
        import mlx_whisper  # lazy: keep import cheap
        return mlx_whisper.transcribe(pcm.astype(np.float32, copy=False), path_or_hf_repo=MODEL, language="en")["text"].strip()
    except Exception:
        return ""


def warmup() -> None:
    transcribe(np.zeros(16_000, np.float32))
