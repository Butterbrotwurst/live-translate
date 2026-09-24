"""Download every model the UI loads on "Start", with a progress bar, and convert the
Albanian Whisper to MLX. Called by Start.command; safe to run again at any time.

A finished download leaves a stamp in models/.stamps and is skipped next time (so a
second start works offline), but a stamp only counts while its files still exist.
Without a stamp, snapshot_download runs again -- instant if the files are already in
the HF cache, and an interrupted download resumes where it stopped.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.utils import filter_repo_objects
from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn, TransferSpeedColumn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from live_translate.config import HELPER_TRANSLATOR, MODELS_DIR, PROFILES, TRANSLATORS, TTS_MODEL, default_translator  # noqa: E402

STAMPS = MODELS_DIR / ".stamps"
SQ_REPO = "Flutra/whisper-large-v3-turbo-sq-v2"
console = Console()


def repo_dir(repo: str) -> Path:
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


def stamped(repo: str) -> bool:
    stamp = STAMPS / repo.replace("/", "--")
    if stamp.exists() and repo_dir(repo).exists():
        return True
    stamp.unlink(missing_ok=True)
    return False


def download(repo: str, label: str, allow: list[str] | None = None) -> None:
    total = None
    try:
        info = HfApi().model_info(repo, files_metadata=True)
        names = [s.rfilename for s in info.siblings or []]
        if allow:
            names = list(filter_repo_objects(names, allow_patterns=allow))
        sizes = {s.rfilename: s.size or 0 for s in info.siblings or []}
        total = sum(sizes.get(n, 0) for n in names) or None
    except Exception:  # offline or API change: the bar just has no end
        pass

    folder = repo_dir(repo)
    columns = (TextColumn("  {task.description}"), BarColumn(bar_width=28), DownloadColumn(), TransferSpeedColumn(), TimeRemainingColumn())
    with Progress(*columns, console=console) as bar:
        task = bar.add_task(label, total=total)
        stop = threading.Event()

        def watch() -> None:
            while not stop.wait(0.5):
                if folder.exists():
                    bar.update(task, completed=min(dir_size(folder), total or float("inf")))

        t = threading.Thread(target=watch, daemon=True)
        t.start()
        try:
            snapshot_download(repo, allow_patterns=allow)
        finally:
            stop.set()
            t.join(timeout=2)
        bar.update(task, completed=total or dir_size(folder))

    STAMPS.mkdir(parents=True, exist_ok=True)
    (STAMPS / repo.replace("/", "--")).touch()


def main() -> int:
    # the translator is picked from this Mac's RAM, so a 16 GB machine gets the 4B one
    wanted = [
        (TRANSLATORS[default_translator()][1], "Übersetzer"),
        (TRANSLATORS[HELPER_TRANSLATOR][1], "Schnellmodell"),
        (TTS_MODEL, "Sprachausgabe"),
        (PROFILES["de"].asr_model, "Spracherkennung Deutsch"),
    ]
    todo = list(dict.fromkeys(repo for repo, _ in wanted if not stamped(repo)))
    sq = Path(PROFILES["sq"].asr_model)
    need_sq = not (sq / "weights.safetensors").exists()
    if not todo and not need_sq:
        console.print("  [dim]alle da[/dim]")
        return 0
    console.print("  [dim]Beim ersten Mal 8–15 GB, je nach Leitung 20–60 Minuten. "
                  "Abbrechen ist ok, beim nächsten Start geht es weiter.[/dim]")
    labels = dict(reversed(wanted))  # first label wins if two roles share a repo
    for repo in todo:
        download(repo, labels[repo])

    if need_sq:
        was_cached = repo_dir(SQ_REPO).exists()
        download(SQ_REPO, "Spracherkennung Albanisch", allow=["*.json", "model.safetensors", "*.txt"])
        console.print("  Albanisch-Modell wird für Apple Silicon umgewandelt …")
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "convert_whisper.py"),
             "--torch-name-or-path", SQ_REPO, "--mlx-path", str(sq), "--dtype", "float16"],
            check=True,
        )
        # the 3 GB PyTorch original is useless once converted -- unless it was there before
        if not was_cached:
            shutil.rmtree(repo_dir(SQ_REPO), ignore_errors=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as e:
        console.print(f"[red]  {type(e).__name__}: {e}[/red]")
        raise SystemExit(1)
