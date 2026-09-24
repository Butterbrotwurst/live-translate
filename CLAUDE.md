# live-translate

Lokales Live-Übersetzungstool (Albanisch/Deutsch → Englisch) für Apple Silicon. Python 3.13 + uv, alles MLX.

- Pipeline: `live_translate/pipeline.py` (Threads: Segmenter → ASR → MT → TTS), Terminal-CLI in `app.py`, Web-UI in `server.py` (FastAPI + WebSocket) mit `static/index.html`. Start: `uv run live-translate-ui`; Browser-Preview über `.claude/launch.json`.
- Modelle/Sprachprofile zentral in `live_translate/config.py`. Albanisch-Whisper liegt lokal in `models/` (gitignored, siehe README für die Konvertierung).
- Einrichtung für Einsteiger: `Start.command` (Doppelklick) installiert uv nach `~/.local/bin`, macht `uv sync`
  und startet den Server. Fehlende Modelle lädt der Server selbst (`live_translate/setup.py`, Stempel in
  `models/.stamps`), Fortschritt per WebSocket (`type: setup`) im Knopf. Bytes werden über einen tqdm-Hook
  gezählt, nicht über die Ordnergröße: das Xet-Backend schreibt große Dateien erst am Ende in den Cache.
  Braucht weder Homebrew noch die Command Line Tools — neue Abhängigkeiten deshalb nur als fertige Wheels.
  `packaging/` baut dagegen die `.app` für reine Nutzer; deren `bootstrap.sh` überschreibt den Code bei
  jedem Start, zum Entwickeln ungeeignet.
- Tests laufen mit `--file data/samples/de_test.wav` (macOS-`say`-Audio), da kein Albanisch-Audio vorliegt.
- Benchmark: `scripts/bench_flores.py`, Daten in `data/flores200_dataset/` (gitignored, Download-URL im Skript-Kommentar / RESEARCH.md).
- Recherche-Grundlage mit allen Modell-Entscheidungen: `RESEARCH.md`.
- Ollama nutzt auf 24 GB kein MLX → immer `mlx-lm` direkt verwenden.
- Packaging-Falle: alles, was eine Bibliothek erst zur Laufzeit nachlädt (spaCy-Modell
  `en_core_web_sm` für Kokoro, HF-Repos), muss in `pyproject.toml` bzw. in
  `packaging/bootstrap.sh` stehen — im fertigen App-Bundle gibt es weder pip/uv noch
  ffmpeg auf dem PATH.
- Audio-Geräte: PortAudio liest die Geräteliste nur beim Initialisieren. `/devices`
  re-initialisiert deshalb (nur solange keine Session läuft), und die UI schickt
  Geräte-**Namen** statt Indizes — Indizes verschieben sich, sobald ein Gerät kommt oder geht.
- espeak-ng (Fallback-G2P von Kokoro) kopiert seinen Datenpfad in einen 160-Byte-Puffer. Ist der Pfad länger,
  nimmt es still einen Pfad vom CI-Rechner und beendet beim ersten Phonemisieren den ganzen Prozess per `exit()`.
  `tts.fix_espeak_data_path()` kopiert die Daten dann unter einen kurzen Pfad — nicht entfernen.
- UI-Design (`static/index.html`): nur dunkel, Schwarz mit Weiß als Akzent, alle Buttons voll gerundet.
  Oben nur der Start-Knopf (mittig, solange kein Text da ist, dann oben) und der Einstellungs-Knopf; alles
  andere gehört ins Einstellungs-Overlay. Start und Stop sind derselbe Knopf (▶ groß in der Mitte, während
  einer Session ■ klein daneben); das Mikrofon erscheint nur während einer Session und schaltet nur stumm.
  Kein Text, der nicht wirklich nötig ist; Animationen dezent.
- Fluss-Ansicht: `paint()` baut die Zeilen aus dem State und gleicht sie wortweise über Zeilen hinweg mit
  dem Bildschirm ab. Streamende Zeilen ändern sich nur an Ort und Stelle und wachsen am Ende — nie
  Wörter mittendrin einfügen, sonst landet der neueste Text in der Mitte. Umsortieren (LCS über Zeilen,
  Wörter gleiten per FLIP) erst, wenn Text final ist. Wort-Animationen nur per Web Animations API —
  CSS-Animationen und -Transitions starten neu, sobald ein Wort in eine andere Zeile wandert. `hold()`
  lässt einen neu startenden, noch kürzeren Stream warten, bis er den alten Text eingeholt hat;
  Platzhalter überbrücken die Lücke zwischen Live-Transkript und finaler Spracherkennung.
- Utterances und Blöcke zählen pro Session ab 1; der Server schickt `sid` mit, die Seite schlüsselt
  `sid.idx`. Nie nur nach `idx` schlüsseln — sonst überschreibt eine neue Session die obersten Zeilen.
- Fluss ist unten verankert (`main` mit `justify-content: flex-end`); wächst er, gleitet die ganze
  Spalte hoch (`flow._lift`). Der Orb unter dem Text ist thinking-orbs (MIT), vendored unter
  `static/vendor/` und über `/static` ausgeliefert — kein CDN, das Tool muss offline laufen.
