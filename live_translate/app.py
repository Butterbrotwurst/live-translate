from __future__ import annotations

import argparse
import threading
import time

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from .config import PROFILES, TRANSLATORS, TTS_VOICE
from .pipeline import Pipeline, Utterance


def build_table(p: Pipeline, status: str, rows: int = 10) -> Table:
    t = Table(expand=True, title=f"live-translate  {p.lang.name} → English   [{p.translator_name}]   {status}")
    t.add_column("#", width=3)
    t.add_column(p.lang.name, ratio=1)
    t.add_column("English", ratio=1, style="bold")
    t.add_column("asr", width=6, justify="right")
    t.add_column("mt", width=6, justify="right")
    t.add_column("tts", width=6, justify="right")
    for u in p.utterances[-rows:]:
        if u.skipped:
            continue
        src = u.source or Text(f"… ({u.audio_s:.1f}s)", style="dim")
        eng = u.english or (Text("…", style="dim") if u.source else "")
        f = lambda ms: f"{ms:.0f}" if ms else ""
        t.add_row(str(u.idx), src, eng, f(u.asr_ms), f(u.mt_ms), f(u.tts_ms))
    return t


def main() -> None:
    ap = argparse.ArgumentParser(prog="live-translate")
    ap.add_argument("--lang", choices=PROFILES, default="de")
    ap.add_argument("--translator", choices=TRANSLATORS, default="qwen")
    ap.add_argument("--no-tts", action="store_true")
    ap.add_argument("--voice", default=TTS_VOICE)
    ap.add_argument("--context", type=int, default=3, help="previous utterances given as context")
    ap.add_argument("--file", help="process an audio file instead of the microphone")
    ap.add_argument("--realtime", action="store_true", help="with --file: feed audio at real-time speed")
    ap.add_argument("--input-device", type=int)
    ap.add_argument("--output-device", type=int)
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return

    console = Console()
    status = {"s": "starting"}
    p = Pipeline(
        lang=PROFILES[args.lang],
        translator_name=args.translator,
        tts_enabled=not args.no_tts,
        voice=args.voice,
        context_turns=args.context,
        output_device=args.output_device,
    )
    p.on_status = lambda s: status.__setitem__("s", s)

    with Live(build_table(p, status["s"]), console=console, refresh_per_second=8, screen=False) as live:
        p.on_status = lambda s: (status.__setitem__("s", s), live.update(build_table(p, s)))
        p.on_update = lambda u: live.update(build_table(p, status["s"]))
        p.load()
        if args.file:
            p.run_file(args.file, realtime=args.realtime)
            live.update(build_table(p, "done"))
            p.close()
        else:
            stop = threading.Event()
            t = threading.Thread(target=p.run_mic, args=(args.input_device, stop), daemon=True)
            t.start()
            try:
                while t.is_alive():
                    time.sleep(0.2)
            except KeyboardInterrupt:
                stop.set()
                t.join()
                p.wait_idle(timeout=30)
                live.update(build_table(p, "stopped"))
                p.close()


if __name__ == "__main__":
    main()
