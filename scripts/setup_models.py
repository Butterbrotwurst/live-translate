"""Download every model the UI needs, with a progress bar in the terminal.

The server does the same on its own at startup (progress on the page), so this is only
for fetching everything ahead of time or without the UI. Logic: live_translate/setup.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn, TransferSpeedColumn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from live_translate import setup  # noqa: E402

console = Console()


def main() -> int:
    if not setup.todo():
        console.print("  [dim]alle da[/dim]")
        return 0
    columns = (TextColumn("  {task.description}"), BarColumn(bar_width=28), DownloadColumn(), TransferSpeedColumn(), TimeRemainingColumn())
    with Progress(*columns, console=console) as bar:
        task = bar.add_task("Prüfe …", total=None)

        def report(p: setup.Progress) -> None:
            if p.phase == "download":
                bar.update(task, description=p.label, completed=p.done, total=p.total or None)
            elif p.phase == "convert":
                bar.update(task, description="Albanisch-Modell wird umgewandelt …", completed=p.total)

        setup.run(report)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as e:
        console.print(f"[red]  {e}[/red]")
        raise SystemExit(1)
