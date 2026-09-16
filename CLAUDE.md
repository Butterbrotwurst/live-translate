# live-translate

Lokales Live-Übersetzungstool (Albanisch/Deutsch → Englisch) für Apple Silicon. Python 3.13 + uv, alles MLX.

- Pipeline: `live_translate/pipeline.py` (Threads: Segmenter → ASR → MT → TTS), Terminal-CLI in `app.py`, Web-UI in `server.py` (FastAPI + WebSocket) mit `static/index.html`. Start: `uv run live-translate-ui`; Browser-Preview über `.claude/launch.json`.
- Modelle/Sprachprofile zentral in `live_translate/config.py`. Albanisch-Whisper liegt lokal in `models/` (gitignored, siehe README für die Konvertierung).
- Tests laufen mit `--file data/samples/de_test.wav` (macOS-`say`-Audio), da kein Albanisch-Audio vorliegt.
- Benchmark: `scripts/bench_flores.py`, Daten in `data/flores200_dataset/` (gitignored, Download-URL im Skript-Kommentar / RESEARCH.md).
- Recherche-Grundlage mit allen Modell-Entscheidungen: `RESEARCH.md`.
- Ollama nutzt auf 24 GB kein MLX → immer `mlx-lm` direkt verwenden.
