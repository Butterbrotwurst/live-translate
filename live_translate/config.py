from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

SAMPLE_RATE = 16_000
VAD_CHUNK = 512  # samples per VAD step (32 ms @ 16 kHz), required by Silero v5


@dataclass(frozen=True)
class LangProfile:
    code: str  # ISO 639-1 as used by Whisper / TranslateGemma
    name: str
    flores: str  # FLORES-200 code
    asr_model: str  # HF repo id or local path (MLX format)


PROFILES: dict[str, LangProfile] = {
    "sq": LangProfile(
        code="sq",
        name="Albanian",
        flores="als_Latn",
        asr_model=str(MODELS_DIR / "whisper-large-v3-turbo-sq-v2-mlx"),
    ),
    "de": LangProfile(
        code="de",
        name="German",
        flores="deu_Latn",
        asr_model="mlx-community/whisper-large-v3-turbo",
    ),
}

# name -> (kind, repo)
TRANSLATORS: dict[str, tuple[str, str]] = {
    "translategemma": ("translategemma", "mlx-community/translategemma-12b-it-4bit"),
    "qwen": ("chat", "mlx-community/Qwen3.5-9B-MLX-4bit"),
    "qwen-4b": ("chat", "mlx-community/Qwen3.5-4B-4bit"),
    "madlad": ("madlad", "Nextcloud-AI/madlad400-3b-mt-ct2-int8"),
}

# fast helper model for provisional translations and sentence polishing (final translations use the main one)
HELPER_TRANSLATOR = "qwen-4b"
POLISH_WITH_MAIN = True  # sentence polishing with the main model (quality) instead of the helper (speed)



def total_ram_gb() -> float:
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3
    except (ValueError, OSError):  # not available -> assume a small machine
        return 16.0


def default_translator() -> str:
    """Biggest translator that comfortably fits alongside ASR + TTS in unified memory."""
    gb = total_ram_gb()
    if gb >= 30:
        return "translategemma"
    if gb >= 22:
        return "qwen"
    return "qwen-4b"


TTS_MODEL = "mlx-community/Kokoro-82M-bf16"
TTS_VOICE = "af_heart"
