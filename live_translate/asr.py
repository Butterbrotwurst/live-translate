from __future__ import annotations

import re
import time

import mlx_whisper
import numpy as np

# Phrases Whisper tends to emit on silence/noise (German + Albanian fine-tune + English)
_HALLUCINATIONS = {
    "vielen dank", "danke", "danke schön", "untertitel im auftrag des zdf", "untertitelung des zdf",
    "untertitel von stephanie geiges", "das war's", "tschüss", "bis zum nächsten mal",
    "thank you", "thanks for watching", "faleminderit", "ju faleminderit", "mirupafshim",
}
_NORM = re.compile(r"[^\w\s]")


def _norm(text: str) -> str:
    return _NORM.sub("", text.lower()).strip()


def looks_hallucinated(text: str, audio_s: float) -> bool:
    """Short stock phrase on a short segment, or a degenerate repetition -> hallucination."""
    t = _norm(text)
    if t in _HALLUCINATIONS and audio_s <= 4.0:
        return True
    words = t.split()
    return len(words) >= 6 and len(set(words)) <= 2  # "ja ja ja ja ja ja"


def similar(a: str, b: str) -> float:
    """Word-set Jaccard similarity, used to drop Whisper repeating the previous utterance on noise."""
    wa, wb = set(_norm(a).split()), set(_norm(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


class ASR:
    def __init__(self, model_path: str, language: str, min_rms: float = 0.004, use_prompt: bool = False):
        self.model_path = model_path
        self.language = language
        self.min_rms = min_rms
        self.use_prompt = use_prompt
        self.last_text = ""

    def warmup(self) -> None:
        self.transcribe(np.zeros(16_000, dtype=np.float32), mode="warmup")

    def transcribe(self, audio: np.ndarray, mode: str = "final") -> tuple[str, float]:
        """mode: "final" (filters + remembers text), "partial" (live preview, no dedupe, does not
        remember text - otherwise the final segment would look like an echo of its own preview), "warmup"."""
        t0 = time.perf_counter()
        if mode != "warmup" and float(np.sqrt(np.mean(audio**2))) < self.min_rms:
            return "", time.perf_counter() - t0
        r = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model_path,
            language=self.language,
            task="transcribe",
            temperature=0.0,  # no temperature fallback: it mostly produces hallucinations on noise
            condition_on_previous_text=False,
            # previous final text as prompt: fragments starting mid-sentence get continuation-style
            # casing/punctuation instead of being guessed as fresh sentences or questions
            initial_prompt=(self.last_text[-200:] or None) if mode != "warmup" and self.use_prompt else None,
            no_speech_threshold=0.5,
            compression_ratio_threshold=2.2,
            logprob_threshold=None,
            fp16=True,
            verbose=None,
        )
        parts = []
        for s in r.get("segments", []):
            txt = s["text"].strip()
            if not txt:
                continue
            if s.get("no_speech_prob", 0.0) > 0.5 or s.get("avg_logprob", 0.0) < -1.2:
                continue  # Whisper itself is unsure there was speech
            if s.get("compression_ratio", 0.0) > 2.2:
                continue  # repetitive garbage
            parts.append(txt)
        text = " ".join(parts)
        if mode != "warmup" and looks_hallucinated(text, len(audio) / 16_000):
            text = ""
        if mode == "final":
            if text and self.last_text and similar(text, self.last_text) > 0.8:
                text = ""  # same content as the previous utterance -> noise echo
            if text:
                self.last_text = text
        return text, time.perf_counter() - t0
