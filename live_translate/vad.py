from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from silero_vad import load_silero_vad

from .config import SAMPLE_RATE, VAD_CHUNK

CHUNK_S = VAD_CHUNK / SAMPLE_RATE


@dataclass
class Segment:
    audio: np.ndarray
    speech_ratio: float  # fraction of chunks the VAD considered speech
    rms: float
    end_reason: str = "silence"  # silence | soft (short pause after soft_max) | hard (time limit) | flush


class Segmenter:
    """Turns a stream of 512-sample float32 chunks into speech segments.

    Rules (seconds of audio):
      * speech must be detected for `min_speech_s` before a segment is opened (rejects clicks/noise blips)
      * `preroll_s` of audio before the trigger is prepended
      * a segment ends after `min_silence_s` of silence
      * once longer than `soft_max_s`, a short pause of `soft_pause_s` is enough to end it
      * at `hard_max_s` it is cut regardless (speaker never pauses)
    """

    def __init__(
        self,
        on: float = 0.6,
        off: float = 0.35,
        min_speech_s: float = 0.25,
        min_silence_s: float = 0.4,
        soft_max_s: float = 2.0,
        soft_pause_s: float = 0.15,
        hard_max_s: float = 3.5,
        preroll_s: float = 0.3,
        min_segment_s: float = 0.4,
    ):
        self.model = load_silero_vad(onnx=True)
        self.on, self.off = on, off
        self.min_speech_chunks = max(1, int(min_speech_s / CHUNK_S))
        self.min_silence_s, self.soft_max_s, self.soft_pause_s, self.hard_max_s = (
            min_silence_s, soft_max_s, soft_pause_s, hard_max_s,
        )
        self.min_segment_s = min_segment_s
        self.preroll: deque[np.ndarray] = deque(maxlen=max(1, int(preroll_s / CHUNK_S)))
        self.buf: list[np.ndarray] = []
        self.probs: list[float] = []
        self.speech_chunks = 0
        self.in_speech = False  # confirmed speech
        self.pending = 0  # consecutive speech chunks before confirmation
        self.silence_s = 0.0

    def prob(self, chunk: np.ndarray) -> float:
        return float(self.model(torch.from_numpy(chunk), SAMPLE_RATE).item())

    def _make(self, reason: str) -> Segment:
        audio = np.concatenate(self.buf)
        return Segment(audio, min(1.0, self.speech_chunks / max(1, len(self.buf))), float(np.sqrt(np.mean(audio**2))), reason)

    def feed(self, chunk: np.ndarray) -> list[Segment]:
        assert chunk.shape == (VAD_CHUNK,), chunk.shape
        p = self.prob(chunk)
        out: list[Segment] = []

        if not self.in_speech:
            self.preroll.append(chunk)
            if p > self.on:
                self.pending += 1
                if self.pending >= self.min_speech_chunks:
                    self.in_speech = True
                    self.buf = list(self.preroll)
                    self.probs = [1.0] * len(self.buf)
                    self.speech_chunks = self.pending
                    self.silence_s = 0.0
                    self.pending = 0
            else:
                self.pending = 0
            return out

        self.buf.append(chunk)
        self.probs.append(p)
        if p > self.on:
            self.speech_chunks += 1
        self.silence_s = self.silence_s + CHUNK_S if p < self.off else 0.0
        length_s = len(self.buf) * CHUNK_S

        real_pause = self.silence_s >= self.min_silence_s
        end = real_pause or (length_s >= self.soft_max_s and self.silence_s >= self.soft_pause_s)
        if end:
            self.in_speech = False
            self.model.reset_states()
            if length_s - self.silence_s >= self.min_segment_s:
                out.append(self._make("silence" if real_pause else "soft"))
            self.buf, self.probs = [], []
            self.speech_chunks = 0
        elif length_s >= self.hard_max_s:
            # cut at the quietest chunk of the last 1.5 s instead of mid-word at the exact limit
            window = max(1, int(1.5 / CHUNK_S))
            tail = self.probs[-window:]
            cut = len(self.buf) - window + int(np.argmin(tail))
            rest, rest_p = self.buf[cut:], self.probs[cut:]
            self.buf, self.probs = self.buf[:cut], self.probs[:cut]
            out.append(self._make("hard"))
            self.buf, self.probs = rest, rest_p  # stay in speech, next segment continues
            self.speech_chunks = sum(1 for q in rest_p if q > self.on)
        return out

    def flush(self) -> list[Segment]:
        out = []
        if self.buf and len(self.buf) * CHUNK_S >= self.min_segment_s:
            out.append(self._make("flush"))
        self.buf, self.probs = [], []
        self.in_speech = False
        self.pending = 0
        self.model.reset_states()
        return out
