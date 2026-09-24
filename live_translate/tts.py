from __future__ import annotations

import queue
import shutil
import tempfile
import threading
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import sounddevice as sd

from .config import TTS_MODEL, TTS_VOICE

# espeak-ng copies its data path into a fixed 160-byte buffer (N_PATH_HOME). A longer path is
# silently swapped for the one compiled into the wheel -- a CI runner's -- and the first
# phonemization then calls exit(): the whole server dies without a Python traceback. The data
# sits inside .venv, so a project folder deeper than ~90 characters (iCloud Drive, a long user
# name) is enough. phonemizer resolve()s the path, so a symlink does not help; a copy does.
ESPEAK_PATH_MAX = 159


def fix_espeak_data_path() -> None:
    from misaki import espeak  # noqa: F401 -- points EspeakWrapper at the wheel's data on import
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    src = Path(EspeakWrapper._ESPEAK_DATA_PATH or "").resolve()
    if len(str(src)) <= ESPEAK_PATH_MAX:
        return
    name = f"espeak-ng-data-{version('espeakng-loader')}"
    homes = [Path.home() / "Library" / "Caches" / "live-translate", Path(tempfile.gettempdir()) / "live-translate"]
    dst = next((h.resolve() / name for h in homes if len(str(h.resolve() / name)) <= ESPEAK_PATH_MAX), None)
    if dst is None:
        raise RuntimeError(f"Pfad zu lang für espeak-ng ({len(str(src))} Zeichen): Projekt in einen kürzeren Ordner verschieben.")
    if not (dst / "phontab").exists():
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(src, tmp)
        shutil.rmtree(dst, ignore_errors=True)
        tmp.rename(dst)
    EspeakWrapper.set_data_path(str(dst))


class TTS:
    def __init__(self, repo: str = TTS_MODEL, voice: str = TTS_VOICE, speed: float = 1.0):
        from mlx_audio.tts.utils import load_model

        fix_espeak_data_path()
        self.model = load_model(repo)
        self.voice = voice
        self.speed = speed

    def synth(self, text: str) -> tuple[np.ndarray, int, float]:
        t0 = time.perf_counter()
        chunks, sr = [], 24_000
        for r in self.model.generate(text, voice=self.voice, speed=self.speed, lang_code="a"):
            chunks.append(np.asarray(r.audio, dtype=np.float32).reshape(-1))
            sr = r.sample_rate
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        return audio, sr, time.perf_counter() - t0


class Player:
    """Plays audio clips sequentially on a background thread through one persistent output stream."""

    def __init__(self, device: int | None = None, sample_rate: int = 24_000):
        self.q: queue.Queue[tuple[np.ndarray, int] | None] = queue.Queue()
        self.device = device
        self.sample_rate = sample_rate
        self.stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32", device=device)
        self.stream.start()
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def play(self, audio: np.ndarray, sr: int) -> None:
        self.q.put((audio, sr))

    def _run(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                break
            audio, sr = item
            if sr != self.sample_rate:  # simple linear resample; Kokoro is 24 kHz anyway
                n = int(len(audio) * self.sample_rate / sr)
                audio = np.interp(np.linspace(0, len(audio), n, endpoint=False), np.arange(len(audio)), audio)
            if audio.size:
                self.stream.write(np.ascontiguousarray(audio, dtype=np.float32).reshape(-1, 1))
            self.q.task_done()

    def close(self) -> None:
        self.q.put(None)
        self.t.join(timeout=30)
        self.stream.stop()
        self.stream.close()
