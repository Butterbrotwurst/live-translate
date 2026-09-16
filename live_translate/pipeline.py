from __future__ import annotations

import datetime as dt
import itertools
import queue
import re
import threading
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .asr import ASR
from .config import HELPER_TRANSLATOR, POLISH_WITH_MAIN, ROOT, SAMPLE_RATE, VAD_CHUNK, LangProfile
from .translate import History, Translator, make_translator
from .vad import Segment, Segmenter


def _load_audio(path: str) -> np.ndarray:
    """Mono float32 at SAMPLE_RATE. Plain PCM WAV is read with the stdlib, because the
    packaged app runs on Macs without ffmpeg -- which is what mlx_whisper's loader shells
    out to. Anything else still goes through ffmpeg."""
    try:
        with wave.open(path, "rb") as w:
            if w.getsampwidth() != 2 or w.getcomptype() != "NONE":
                raise wave.Error("not 16-bit PCM")
            channels, rate = w.getnchannels(), w.getframerate()
            raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    except (wave.Error, EOFError):
        try:
            from mlx_whisper.audio import load_audio
            return np.array(load_audio(path, SAMPLE_RATE), dtype=np.float32)
        except FileNotFoundError as e:  # ffmpeg missing
            raise RuntimeError(f"{Path(path).name}: nur unkomprimierte WAV-Dateien werden ohne ffmpeg unterstützt") from e
    if channels > 1:
        raw = raw.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        n = int(len(raw) * SAMPLE_RATE / rate)
        raw = np.interp(np.linspace(0, len(raw), n, endpoint=False), np.arange(len(raw)), raw).astype(np.float32)
    return np.ascontiguousarray(raw, dtype=np.float32)


# Loaded models are expensive; keep them across pipeline restarts (switching settings in the UI).
_MODEL_CACHE: dict[tuple, Any] = {}


def _cached(key: tuple, factory: Callable[[], Any]) -> Any:
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = factory()
    return _MODEL_CACHE[key]


@dataclass
class Utterance:
    idx: int
    audio_s: float
    t_end: float  # wall clock (perf_counter) when the segment ended
    t_session: float  # seconds since session start, for the transcript
    source: str = ""
    english: str = ""
    partial_english: str = ""
    asr_ms: float = 0.0
    mt_ms: float = 0.0
    tts_ms: float = 0.0
    latency_ms: float = 0.0  # segment end -> translation shown
    done: bool = False
    skipped: bool = False
    end_reason: str = "silence"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Block:
    """A group of consecutive fragments that the polish stage rewrites as clean sentences."""

    id: int
    t_session: float
    utterance_ids: list[int] = field(default_factory=list)
    source_clean: str = ""
    english_clean: str = ""
    polished_upto: int = 0  # highest utterance idx covered by source_clean/english_clean
    closed: bool = False
    version: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


POLISH_SYSTEM = (
    "You are the editor of a live interpretation at a university lecture. The speech arrives in fragments cut by "
    "pauses or a time limit, so sentences are often split across fragments, and the punctuation or capitalization "
    "at the cut points is unreliable: a fragment may end with a full stop although the sentence continues, or start "
    "lowercase although a new sentence begins. You get OPEN (the last sentence so far, possibly unfinished), NEW "
    "(the translation of the next fragment) and PAUSE (whether the speaker paused between them). Decide whether NEW "
    "continues the sentence in OPEN. Return OPEN and NEW as clean English: one sentence if NEW continues it, "
    "otherwise OPEN properly ended and NEW as a new sentence. Fix punctuation, capitalization and duplicated or "
    "stuttered words at the junction; keep the wording and all content otherwise; never add, drop or summarize; "
    "never finish a sentence the speaker has not finished. Answer in exactly this format:\n"
    "ENGLISH: <clean text>\nJOINED: yes or no\nCOMPLETE: yes or no (does the text end with a finished sentence?)"
)


_SENT_END_RE = re.compile(r"[.!?…][\"'”)]*\s+(?=\S)|[.!?…][\"'”)]*$")


def split_finished(text: str) -> tuple[str, str]:
    """Split into (finished sentences, trailing unfinished sentence)."""
    last = None
    for m in _SENT_END_RE.finditer(text):
        last = m
    if last is None:
        return "", text.strip()
    return text[: last.end()].strip(), text[last.end() :].strip()


@dataclass
class Pipeline:
    lang: LangProfile
    translator_name: str
    tts_enabled: bool = True
    voice: str = "af_heart"
    context_turns: int = 3
    on_update: Callable[[Utterance], None] = lambda u: None
    on_status: Callable[[str], None] = lambda s: None
    on_partial: Callable[[str], None] = lambda s: None  # live transcript of the segment being spoken
    on_provisional: Callable[[str], None] = lambda s: None  # provisional translation of that live transcript
    on_block: Callable[[Block], None] = lambda b: None
    on_level: Callable[[float], None] = lambda v: None  # live input peak, 0..1
    output_device: int | None = None
    partial_every_s: float = 1.2
    provisional: bool = True
    polish: bool = True
    block_max_fragments: int = 8
    block_gap_s: float = 6.0

    utterances: list[Utterance] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    history: History = field(default_factory=list)
    started_at: dt.datetime | None = None

    # --- lifecycle -------------------------------------------------------
    def load(self) -> None:
        self.on_status(f"loading ASR …")
        self.asr: ASR = _cached(("asr", self.lang.asr_model, self.lang.code), lambda: ASR(self.lang.asr_model, self.lang.code))
        self.asr.warmup()
        self.on_status(f"loading translator {self.translator_name} …")
        self.translator: Translator = _cached(
            ("mt", self.translator_name), lambda: make_translator(self.translator_name, self.context_turns)
        )
        if hasattr(self.translator, "context_turns"):
            self.translator.context_turns = self.context_turns  # type: ignore[attr-defined]
        self.translator.translate("Hello.", self.lang.code, [])
        self.helper: Translator = self.translator
        if (self.provisional or self.polish) and HELPER_TRANSLATOR != self.translator_name and hasattr(self.translator, "complete"):
            self.on_status("loading helper model …")
            self.helper = _cached(("mt", HELPER_TRANSLATOR), lambda: make_translator(HELPER_TRANSLATOR, self.context_turns))
            self.helper.translate("Hello.", self.lang.code, [])
        self.polish = self.polish and hasattr(self.helper, "complete")
        self.polisher: Translator = self.translator if POLISH_WITH_MAIN and hasattr(self.translator, "complete") else self.helper
        self.tts = None
        if self.tts_enabled:
            from .tts import TTS, Player

            self.on_status("loading TTS …")
            self.tts = _cached(("tts", self.voice), lambda: TTS(voice=self.voice))
            self.tts.synth("Ready.")
            self.player = Player(device=self.output_device)
        self.seg_q: queue.Queue = queue.Queue()
        self.asr_q: queue.PriorityQueue = queue.PriorityQueue()  # (priority, n, payload): finals before provisionals
        self._n = itertools.count()
        self._latest_prov: tuple[int, str] = (0, "")
        self.tts_q: queue.Queue = queue.Queue()
        self.segmenter = Segmenter()
        self._t0 = time.perf_counter()
        self._last_partial = 0.0
        self._live_seq = 0
        self.started_at = dt.datetime.now()
        self._threads = [
            threading.Thread(target=self._asr_worker, daemon=True),
            threading.Thread(target=self._mt_worker, daemon=True),
            threading.Thread(target=self._tts_worker, daemon=True),
        ]
        for t in self._threads:
            t.start()
        self.on_status("ready")

    def close(self) -> None:
        self.seg_q.put(None)
        self.asr_q.put((-1, next(self._n), None))
        self.tts_q.put(None)
        if self.tts is not None:
            self.player.close()

    # --- input -----------------------------------------------------------
    def feed_chunk(self, chunk: np.ndarray) -> None:
        was_speaking = self.segmenter.in_speech
        segs = self.segmenter.feed(chunk)
        for seg in segs:
            self._emit(seg)
        if self.segmenter.in_speech != was_speaking or segs:
            self.on_status("speaking" if self.segmenter.in_speech else "listening")
            if not self.segmenter.in_speech or segs:
                self._live_seq += 1
                self.on_partial("")
                self.on_provisional("")
        # live transcript of the ongoing segment
        if self.segmenter.in_speech and self.segmenter.buf:
            now = time.perf_counter()
            buffered = len(self.segmenter.buf) * VAD_CHUNK / SAMPLE_RATE
            if buffered >= self.partial_every_s and now - self._last_partial >= self.partial_every_s:
                self._last_partial = now
                self.seg_q.put(("partial", self._live_seq, np.concatenate(self.segmenter.buf)))

    def flush(self) -> None:
        for seg in self.segmenter.flush():
            self._emit(seg)
        self._live_seq += 1
        self.on_partial("")
        self.on_provisional("")
        self.on_status("listening")

    def _emit(self, seg: Segment) -> None:
        if seg.speech_ratio < 0.3:  # mostly noise / breathing / background
            return
        now = time.perf_counter()
        u = Utterance(
            idx=len(self.utterances) + 1, audio_s=len(seg.audio) / SAMPLE_RATE, t_end=now, t_session=now - self._t0,
            end_reason=seg.end_reason,
        )
        self.utterances.append(u)
        self.on_update(u)
        self.seg_q.put(("final", u, seg.audio))

    def wait_idle(self, timeout: float = 120.0) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if all(u.done or u.skipped for u in self.utterances) and self.seg_q.empty() and self.asr_q.empty():
                if self.tts is None or (self.tts_q.empty() and self.player.q.unfinished_tasks == 0):
                    return
            time.sleep(0.1)

    # --- workers ---------------------------------------------------------
    def _asr_worker(self) -> None:
        while True:
            item = self.seg_q.get()
            if item is None:
                return
            kind, a, seg = item
            if kind == "partial":
                if a != self._live_seq:  # stale: the segment has already ended
                    continue
                text, _ = self.asr.transcribe(seg, mode="partial")
                if a == self._live_seq and text:
                    self.on_partial(text)
                    if self.provisional:
                        self._latest_prov = (a, text)
                        self.asr_q.put((2, next(self._n), ("prov", a, text)))
                continue
            u: Utterance = a
            text, dt_ = self.asr.transcribe(seg)
            u.asr_ms = dt_ * 1000
            u.source = text
            if not text or len(text.strip(" .,!?")) < 2:
                u.skipped = True
                self.on_update(u)
                continue
            self.on_update(u)
            self.asr_q.put((0, next(self._n), u))

    def _final_pending(self) -> bool:
        with self.asr_q.mutex:
            return any(item[0] == 0 for item in self.asr_q.queue)

    def _mt_worker(self) -> None:
        while True:
            _, _, u = self.asr_q.get()
            if u is None:
                return
            if isinstance(u, tuple) and u[0] == "polish":
                self._polish(u[1], u[2])
                continue
            if isinstance(u, tuple):  # provisional translation of the live transcript
                _, seq, text = u
                if (seq, text) != self._latest_prov or seq != self._live_seq or self._final_pending():
                    continue  # stale, or a final segment is waiting: skip
                last_p = [0.0]

                def prov_partial(t: str, seq=seq) -> None:
                    now = time.perf_counter()
                    if now - last_p[0] > 0.08 and seq == self._live_seq:
                        last_p[0] = now
                        self.on_provisional(t)

                english, _ = self.helper.translate(
                    text, self.lang.code, self.history, on_partial=prov_partial,
                    should_stop=lambda seq=seq: seq != self._live_seq or self._final_pending(),
                )
                if seq == self._live_seq and english:
                    self.on_provisional(english)
                continue
            blk = self._assign_block(u)
            last = [0.0]

            def partial(text: str, u=u, last=last) -> None:
                now = time.perf_counter()
                if now - last[0] > 0.08:
                    last[0] = now
                    u.partial_english = text
                    self.on_update(u)

            english, dt_ = self.translator.translate(u.source, self.lang.code, self.history, on_partial=partial)
            u.mt_ms = dt_ * 1000
            u.english = english
            u.partial_english = ""
            u.latency_ms = (time.perf_counter() - u.t_end) * 1000
            self.history.append((u.source, english))
            self.history = self.history[-10:]
            if self.tts is None:
                u.done = True
            self.on_update(u)
            if self.tts is not None:
                self.tts_q.put(u)
            if self.polish and blk is not None:
                blk.source_clean = self._join_source(blk)
                blk.version += 1
                self.on_block(blk)
                self.asr_q.put((1, next(self._n), ("polish", blk.id, blk.version)))

    # --- blocks / polish ---------------------------------------------------
    def _assign_block(self, u: Utterance) -> Block | None:
        if not self.polish:
            return None
        blk = self.blocks[-1] if self.blocks else None
        if (
            blk is None
            or blk.closed
            or len(blk.utterance_ids) >= self.block_max_fragments
            or u.t_session - self._utt(blk.utterance_ids[-1]).t_session - self._utt(blk.utterance_ids[-1]).audio_s > self.block_gap_s
        ):
            if blk is not None and not blk.closed:
                blk.closed = True
                self.on_block(blk)
            blk = Block(id=len(self.blocks) + 1, t_session=u.t_session - u.audio_s)
            self.blocks.append(blk)
        blk.utterance_ids.append(u.idx)
        return blk

    def _join_source(self, blk: Block) -> str:
        parts = []
        for k, i in enumerate(blk.utterance_ids):
            u = self._utt(i)
            if not u.source:
                continue
            txt = u.source.strip()
            nxt = self._utt(blk.utterance_ids[k + 1]) if k + 1 < len(blk.utterance_ids) else None
            if nxt is not None and u.end_reason in ("soft", "hard"):
                txt = txt.rstrip(".?!…")  # cut mid-sentence: Whisper's sentence-final punctuation is a guess
            parts.append(txt)
        return " ".join(parts)

    def _utt(self, idx: int) -> Utterance:
        return self.utterances[idx - 1]

    def _polish(self, block_id: int, version: int) -> None:
        blk = self.blocks[block_id - 1]
        if blk.version != version:
            return  # a newer fragment was added; that job will cover it
        new = [self._utt(i) for i in blk.utterance_ids if i > blk.polished_upto and self._utt(i).english]
        if not new:
            return
        finished, open_text = split_finished(blk.english_clean)
        if blk.polished_upto and not open_text:
            # always hand the editor the last sentence: its final punctuation may be Whisper's guess
            finished, open_text = split_finished(blk.english_clean.rstrip(" .!?…"))
            open_text = blk.english_clean[len(finished):].strip()
        paused_before = self._utt(blk.polished_upto).end_reason in ("silence", "flush") if blk.polished_upto else True
        user = f"OPEN: {open_text}\nNEW: {' '.join(u.english for u in new)}\nPAUSE: {'yes' if paused_before else 'no'}"
        out = self.polisher.complete(  # type: ignore[attr-defined]
            POLISH_SYSTEM, user, max_tokens=220,
            should_stop=lambda: self._final_pending() or blk.version != version,
        )
        if out is None:
            if blk.version == version:  # aborted for a final segment: try again later
                self.asr_q.put((1, next(self._n), ("polish", blk.id, version)))
            return
        merged, complete = None, False
        for line in out.splitlines():
            low = line.strip().lower()
            if low.startswith("english:"):
                merged = line.split(":", 1)[1].strip()
            elif low.startswith("complete:"):
                complete = "yes" in low
        if not merged:  # unparseable answer: fall back to the raw join
            merged = " ".join([open_text] + [u.english for u in new]).strip()
            complete = merged.endswith((".", "!", "?"))
        if blk.version != version:
            return
        blk.english_clean = (finished + " " + merged).strip()
        blk.source_clean = self._join_source(blk)
        blk.polished_upto = new[-1].idx
        # close only when the sentence is complete AND the speaker actually paused; fragments cut by the
        # time limit or a tiny pause are usually mid-sentence even if Whisper put a full stop there
        paused = new[-1].end_reason in ("silence", "flush")
        if (complete and paused) or len(blk.utterance_ids) >= self.block_max_fragments:
            blk.closed = True
        self.on_block(blk)

    def _tts_worker(self) -> None:
        while True:
            u = self.tts_q.get()
            if u is None:
                return
            audio, sr, dt_ = self.tts.synth(u.english)  # type: ignore[union-attr]
            u.tts_ms = dt_ * 1000
            self.player.play(audio, sr)
            u.done = True
            self.on_update(u)

    # --- sources ---------------------------------------------------------
    # A microphone macOS has not granted access to still opens and still delivers
    # chunks -- they are just all exactly zero. Without this check the app sits on
    # "Hört zu" forever and looks broken. A real mic always has a noise floor.
    SILENCE_LIMIT_S = 5.0

    def run_mic(self, device: int | None, stop: threading.Event) -> None:
        import sounddevice as sd

        audio_q: queue.Queue[np.ndarray] = queue.Queue()

        def cb(indata, frames, t, status):
            audio_q.put(indata[:, 0].copy())

        silent_s = 0.0
        heard_anything = False
        last_level = 0.0
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=VAD_CHUNK, device=device, callback=cb
        ):
            self.on_status("listening")
            while not stop.is_set():
                try:
                    chunk = audio_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                peak = float(np.max(np.abs(chunk))) if chunk.size else 0.0
                if peak > 0.0:
                    heard_anything = True
                elif not heard_anything:
                    silent_s += len(chunk) / SAMPLE_RATE
                    if silent_s >= self.SILENCE_LIMIT_S:
                        raise RuntimeError(
                            "Kein Mikrofonsignal. Bitte in den Systemeinstellungen unter "
                            "„Datenschutz & Sicherheit → Mikrofon“ den Zugriff für Live Translate erlauben."
                        )
                # a coarse level, so the UI can show that something is actually arriving
                level = round(min(peak, 1.0), 3)
                if abs(level - last_level) > 0.02:
                    last_level = level
                    self.on_level(level)
                self.feed_chunk(chunk)
        self.flush()

    def run_file(self, path: str, realtime: bool = False, stop: threading.Event | None = None) -> None:
        audio = _load_audio(path)
        n = len(audio) // VAD_CHUNK * VAD_CHUNK
        self.on_status("listening")
        for i in range(0, n, VAD_CHUNK):
            if stop is not None and stop.is_set():
                break
            self.feed_chunk(audio[i : i + VAD_CHUNK])
            if realtime:
                time.sleep(VAD_CHUNK / SAMPLE_RATE)
        self.flush()
        self.wait_idle()

    # --- transcript ------------------------------------------------------
    def transcript_markdown(self) -> str:
        start = self.started_at or dt.datetime.now()
        lines = [
            f"# Transkript {start:%d.%m.%Y %H:%M}",
            "",
            f"- Sprache: {self.lang.name} → English",
            f"- ASR: `{self.lang.asr_model}` · Übersetzer: `{self.translator_name}`",
            "",
        ]
        for eng, src, t in self._clean_rows():
            m, s = divmod(int(t), 60)
            lines.append(f"**[{m:02d}:{s:02d}]** {eng}  ")
            lines.append(f"<sub>{src}</sub>")
            lines.append("")
        return "\n".join(lines)

    def _clean_rows(self) -> list[tuple[str, str, float]]:
        """(english, source, t_session) per block, polished text plus any not-yet-polished tail."""
        rows = []
        covered: set[int] = set()
        for b in self.blocks:
            tail = [self._utt(i) for i in b.utterance_ids if i > b.polished_upto and self._utt(i).english]
            eng = " ".join(x for x in [b.english_clean] + [u.english for u in tail] if x)
            src = b.source_clean or " ".join(self._utt(i).source for i in b.utterance_ids)
            covered.update(b.utterance_ids)
            if eng:
                rows.append((eng, src, b.t_session))
        for u in self.utterances:  # utterances never assigned to a block (polish off)
            if u.idx not in covered and u.english and not u.skipped:
                rows.append((u.english, u.source, u.t_session))
        rows.sort(key=lambda r: r[2])
        return rows

    def transcript_text(self, english_only: bool = False) -> str:
        out = []
        for eng, src, t in self._clean_rows():
            m, s = divmod(int(t), 60)
            out.append(f"[{m:02d}:{s:02d}] {eng}" + ("" if english_only else f"\n         {src}"))
        return "\n".join(out) + "\n"

    def save_transcript(self, directory: Path | None = None) -> Path | None:
        if not any(u.english for u in self.utterances):
            return None
        directory = directory or ROOT / "transcripts"
        directory.mkdir(parents=True, exist_ok=True)
        start = self.started_at or dt.datetime.now()
        path = directory / f"{start:%Y-%m-%d_%H-%M}_{self.lang.code}-en.md"
        path.write_text(self.transcript_markdown())
        return path
