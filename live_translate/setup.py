"""Model setup: every model the UI loads on "Start", downloaded once, plus the Albanian
Whisper converted to MLX. The server runs this at startup and streams the progress to the
page; scripts/setup_models.py does the same with a progress bar in the terminal.

A finished download leaves a stamp in models/.stamps and is skipped next time (so later
starts work offline), but a stamp only counts while its files still exist. Without one,
every file is checked again -- instant if it is already in the HF cache, and an
interrupted download resumes where it stopped.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import HELPER_TRANSLATOR, MODELS_DIR, PROFILES, ROOT, TRANSLATORS, TTS_MODEL, default_translator

STAMPS = MODELS_DIR / ".stamps"
SQ_REPO = "Flutra/whisper-large-v3-turbo-sq-v2"
SQ_FILES = ["*.json", "model.safetensors", "*.txt"]
SQ_MLX = Path(PROFILES["sq"].asr_model)
SQ_MLX_BYTES = 1_700_000_000  # the converted float16 weights, written next to the download
MARGIN = 2_000_000_000  # free space left over after everything is in place


@dataclass
class Progress:
    phase: str = "check"  # check | download | convert | done | error
    done: int = 0  # bytes downloaded or found in the cache, over every repo this run fetches
    total: int = 0  # 0 = unknown (size lookup failed)
    label: str = ""
    eta_s: float | None = None
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def repo_dir(repo: str) -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))


def dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def gb(n: float) -> str:
    return f"{n / 1e9:.1f} GB".replace(".", ",")


def wanted() -> list[tuple[str, str]]:
    """(repo, label) of every HF repo "Start" loads. The translator follows this Mac's RAM,
    so a 16 GB machine gets the 4B model -- which is also the helper, hence the dedupe."""
    pairs = [
        (TRANSLATORS[default_translator()][1], "Übersetzer"),
        (TRANSLATORS[HELPER_TRANSLATOR][1], "Schnellmodell"),
        (TTS_MODEL, "Sprachausgabe"),
        (PROFILES["de"].asr_model, "Spracherkennung Deutsch"),
    ]
    seen: dict[str, str] = {}
    for repo, label in pairs:
        seen.setdefault(repo, label)
    return list(seen.items())


def stamped(repo: str) -> bool:
    stamp = STAMPS / repo.replace("/", "--")
    if stamp.exists() and repo_dir(repo).exists():
        return True
    stamp.unlink(missing_ok=True)
    return False


def todo() -> list[tuple[str, str, list[str] | None]]:
    """What is still missing, as (repo, label, allow_patterns). Only stats files -- cheap."""
    items: list[tuple[str, str, list[str] | None]] = [(r, label, None) for r, label in wanted() if not stamped(r)]
    if not (SQ_MLX / "weights.safetensors").exists():
        items.append((SQ_REPO, "Spracherkennung Albanisch", SQ_FILES))
    return items


def _repo_files(repo: str, allow: list[str] | None) -> list[tuple[str, int]]:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import filter_repo_objects

    info = HfApi().model_info(repo, files_metadata=True)
    sizes = {s.rfilename: s.size or 0 for s in info.siblings or []}
    names = list(filter_repo_objects(sizes, allow_patterns=allow)) if allow else list(sizes)
    return [(n, sizes[n]) for n in names]


def _counting_bar(count: Callable[[int], None]) -> type:
    """A silent tqdm that forwards every byte to `count`. hf_hub_download reports through it
    in real time; the repo folder alone would not show it, because the Xet backend writes a
    big file into the cache only once it is complete."""
    from tqdm.auto import tqdm

    class Bar(tqdm):
        def __init__(self, *args, **kwargs):
            count(kwargs.get("initial") or 0)  # bytes an interrupted download already had
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            count(n or 0)
            return super().update(n)

    return Bar


def _fetch(repo: str, label: str, allow: list[str] | None, files: list[tuple[str, int]] | None, p: Progress) -> bool:
    """True if every file is verified online; False if only an offline copy was found."""
    from huggingface_hub import hf_hub_download, snapshot_download

    if files is None:  # the size lookup failed: offline, but maybe complete from an earlier run
        try:
            snapshot_download(repo, allow_patterns=allow, local_files_only=True)
            return False
        except Exception:
            raise RuntimeError(f"{label} konnte nicht geladen werden. Internetverbindung prüfen.") from None

    lock = threading.Lock()

    def one(name: str, size: int) -> None:
        got = 0

        def count(n: int) -> None:
            nonlocal got
            with lock:
                n = max(0, min(n, size - got))  # never past the file size, whatever tqdm reports
                got += n
                p.done += n

        hf_hub_download(repo, name, tqdm_class=_counting_bar(count))
        count(size)  # a cached file reports nothing -- it still counts as done

    # 8 at a time, like snapshot_download: Kokoro alone has ~50 small voice files, and one
    # after the other each costs a round trip -- slow, and it drags the ETA to half an hour
    pool = ThreadPoolExecutor(max_workers=8)
    try:
        for f in as_completed([pool.submit(one, n, size) for n, size in files]):
            f.result()
    except Exception as e:
        raise RuntimeError(f"{label} konnte nicht geladen werden. Internetverbindung prüfen.") from e
    finally:
        pool.shutdown(cancel_futures=True)
    return True


def run(report: Callable[[Progress], None]) -> None:
    from huggingface_hub.constants import HF_HUB_CACHE
    from huggingface_hub.utils import disable_progress_bars, enable_progress_bars

    items = todo()
    if not items:
        report(Progress("done"))
        return
    p = Progress("check")
    report(p)

    files: dict[str, list[tuple[str, int]] | None] = {}
    for repo, _, allow in items:
        try:
            files[repo] = _repo_files(repo, allow)
        except Exception:  # offline: no total, _fetch below still looks for a local copy
            files[repo] = None
    size = {r: sum(n for _, n in f) for r, f in files.items() if f}
    p.total = sum(size.values())
    cached = sum(min(dir_size(repo_dir(r)), n) for r, n in size.items())

    need_sq = any(r == SQ_REPO for r, _, _ in items)
    sq_was_cached = repo_dir(SQ_REPO).exists()
    Path(HF_HUB_CACHE).mkdir(parents=True, exist_ok=True)
    needed = p.total - cached + (SQ_MLX_BYTES if need_sq else 0) + MARGIN
    free = shutil.disk_usage(HF_HUB_CACHE).free
    if p.total and free < needed:
        raise RuntimeError(f"Zu wenig Speicherplatz: {gb(free)} frei, etwa {gb(needed)} nötig.")

    # the ETA comes from the last 20 s, so a slow start does not haunt the estimate forever;
    # the first 5 s give no estimate at all, cached files fly by and would promise too much
    samples: deque[tuple[float, int]] = deque(maxlen=40)
    stop = threading.Event()

    def watch() -> None:
        while not stop.wait(0.5):
            samples.append((time.monotonic(), p.done))
            (t0, d0), (t1, d1) = samples[0], samples[-1]
            speed = (d1 - d0) / (t1 - t0) if t1 > t0 else 0
            p.eta_s = (p.total - p.done) / speed if p.total and speed > 0 and len(samples) > 10 else None
            report(p)

    p.phase = "download"
    watcher = threading.Thread(target=watch, daemon=True)
    disable_progress_bars()  # the page and our own bar show progress; tqdm would only spam the log
    watcher.start()
    try:
        for repo, label, allow in items:
            p.label = label
            report(p)
            if _fetch(repo, label, allow, files[repo], p) and repo != SQ_REPO:
                STAMPS.mkdir(parents=True, exist_ok=True)
                (STAMPS / repo.replace("/", "--")).touch()
    finally:
        stop.set()
        watcher.join(timeout=2)
        enable_progress_bars()

    if need_sq:
        p.phase, p.label, p.done, p.eta_s = "convert", "Spracherkennung Albanisch", p.total, None
        report(p)
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "convert_whisper.py"),
             "--torch-name-or-path", SQ_REPO, "--mlx-path", str(SQ_MLX), "--dtype", "float16"],
            check=True,
        )
        # the 3 GB PyTorch original is useless once converted -- unless it was there before
        if not sq_was_cached:
            shutil.rmtree(repo_dir(SQ_REPO), ignore_errors=True)
    report(Progress("done"))
