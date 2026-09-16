"""Download one HF repo into the local cache, reporting progress on stdout.

Protocol (read by the LiveTranslate launcher):
  @@PROG <0.0-1.0>   fraction of this repo downloaded
Everything else on stdout/stderr is treated as log output.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path


def _emit(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def main() -> int:
    ap = argparse.ArgumentParser(prog="prefetch")
    ap.add_argument("repo")
    ap.add_argument("--allow", nargs="*", default=None, help="allow_patterns for snapshot_download")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.utils import filter_repo_objects

    api = HfApi()
    expected = 0
    try:
        info = api.model_info(args.repo, files_metadata=True)
        names = [s.rfilename for s in info.siblings or []]
        if args.allow:
            names = list(filter_repo_objects(names, allow_patterns=args.allow))
        sizes = {s.rfilename: (s.size or 0) for s in info.siblings or []}
        expected = sum(sizes.get(n, 0) for n in names)
    except Exception as exc:  # offline / API change -> fall back to spinner
        print(f"size lookup failed: {exc}", file=sys.stderr)

    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    folder = cache / ("models--" + args.repo.replace("/", "--"))

    done = threading.Event()

    def watch() -> None:
        while not done.wait(0.7):
            if expected <= 0 or not folder.exists():
                continue
            _emit(f"@@PROG {min(_dir_size(folder) / expected, 0.999):.4f}")

    t = threading.Thread(target=watch, daemon=True)
    t.start()
    try:
        path = snapshot_download(args.repo, allow_patterns=args.allow)
    finally:
        done.set()
        t.join(timeout=2)
    _emit("@@PROG 1.0")
    print(f"ready: {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
