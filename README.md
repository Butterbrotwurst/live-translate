# live-translate

Lokale Live-Übersetzung gesprochener Sprache (Albanisch oder Deutsch → Englisch) auf Apple Silicon.
Alles läuft offline auf dem Mac: Spracherkennung, kontextbewusste Übersetzung und Sprachausgabe.

```
Mikrofon → Silero VAD → Whisper (mlx-whisper) → LLM-Übersetzung (mlx-lm) → Kokoro TTS (mlx-audio) → Kopfhörer
```

## Schnellstart

**Voraussetzungen:** Mac mit Apple-Chip (M1 oder neuer), mindestens 16 GB RAM, 12–18 GB frei,
Internet für die Ersteinrichtung. Homebrew, Xcode oder die Command Line Tools braucht es **nicht**.

1. **Holen.** Am einfachsten mit [GitHub Desktop](https://desktop.github.com) (bringt git selbst mit):
   *File → Clone Repository* → `live-translate` → z. B. nach `~/Developer/live-translate`.
2. **Doppelklick auf `Start.command`.** Beim ersten Mal installiert es `uv`, Python 3.13 und alle
   Bibliotheken und lädt die Sprachmodelle (8–15 GB, 20–60 Minuten). Danach öffnet sich der Browser.
   Ab dem zweiten Start dauert das ein paar Sekunden.
3. **Mikrofon erlauben**, wenn macOS fragt (die Freigabe gilt für *Terminal*). Weißer Knopf oder
   Leertaste startet, die Einstellungen sitzen oben rechts.

Das Terminal-Fenster ist der Server: offen lassen, solange übersetzt wird; schließen beendet alles.

<details>
<summary>macOS blockiert <code>Start.command</code> („Apple konnte nicht überprüfen …“)</summary>

Das passiert nur, wenn das Projekt als ZIP aus dem Browser kam (geklonte Dateien sind nicht betroffen).
Einmalig: *Systemeinstellungen → Datenschutz & Sicherheit* → ganz unten **Trotzdem öffnen**, dann
nochmal doppelklicken. Oder im Terminal: `xattr -dr com.apple.quarantine <Projektordner>`.
</details>

### Mit Claude weiterentwickeln

Ordner in der Claude-App (Tab *Code*) oder mit `claude` im Terminal öffnen. `CLAUDE.md` gibt Claude
den Überblick über Architektur, Modelle und Fallstricke, `RESEARCH.md` die Begründungen dahinter.
Die Browser-Vorschau ist in `.claude/launch.json` eingerichtet.

Claude arbeitet mit git. Fehlen die Command Line Tools, fragt macOS beim ersten git-Aufruf, ob sie
installiert werden sollen: **Installieren** klicken, das dauert ein paar Minuten und braucht weder
Xcode noch eine Apple-ID.

### Manuell (ohne `Start.command`)

```bash
uv sync
uv run python scripts/setup_models.py   # alle Modelle + Albanisch-Whisper nach MLX konvertieren
uv run live-translate-ui
```

`scripts/setup_models.py` ist idempotent: fertige Downloads hinterlassen einen Stempel in
`models/.stamps`, abgebrochene setzen beim nächsten Lauf fort. Das albanische Whisper-Fine-Tune
(`Flutra/whisper-large-v3-turbo-sq-v2`) wird dabei nach `models/` konvertiert; von Hand geht das so:

```bash
uv run python scripts/convert_whisper.py \
  --torch-name-or-path Flutra/whisper-large-v3-turbo-sq-v2 \
  --mlx-path models/whisper-large-v3-turbo-sq-v2-mlx --dtype float16
```

PortAudio (in `sounddevice`) und espeak-ng (in `espeakng-loader`) kommen als Wheels mit. `ffmpeg`
braucht nur, wer mit `--file` etwas anderes als unkomprimiertes WAV abspielen will.

## Benutzung

### Web-Oberfläche (empfohlen)

```bash
uv run live-translate-ui
```

Öffnet http://127.0.0.1:8765 im Browser. Solange nichts auf dem Bildschirm steht, sitzt der
Start-Knopf in der Mitte; sobald Text erscheint, gleitet er nach oben. Leertaste startet/stoppt.

- **Fluss** (Standard): links Englisch, rechts das Original, zeilenweise ausgerichtet, jedes neue
  Wort blendet weich ein. Während du sprichst, läuft rechts das Live-Transkript mit und links eine
  **vorläufige Übersetzung** (gedimmt). Nach Segmentende ersetzt die endgültige Übersetzung beides.
  Die endgültige läuft immer mit vollem Segment und Kontext, die vorläufige kostet also keine Genauigkeit.
  Absatz nach mehr als 6 s Pause.
- **Details** (`M`): pro Äußerung Original, Übersetzung, Zeitstempel und Laufzeiten (beim Überfahren).
- Der Halo um den Knopf zeigt den Mikrofonpegel; während Modelle laden, dreht sich ein Ring.
- Sprachausgabe läuft parallel über das gewählte Ausgabegerät (AirPods/Kopfhörer empfohlen, sonst hört
  das Mikrofon die Übersetzung mit).

Alles Weitere steckt im Einstellungs-Overlay (oben rechts, `Esc` schließt): Sprache, Übersetzer,
Geräte, Stimme, Ansicht, Schriftgröße, Export (Markdown, Text, nur Englisch, JSON, Kopieren),
„Demo abspielen“ (spielt das deutsche Testaudio durch die ganze Pipeline) und „Verlauf löschen“.
Beim Stoppen wird das Transkript automatisch als Markdown in `transcripts/` gespeichert.

### Terminal

```bash
# Deutsch → Englisch (Testmodus), Qwen3.5 9B, mit Sprachausgabe
uv run live-translate --lang de

# Albanisch → Englisch mit TranslateGemma als Übersetzer, ohne Sprachausgabe
uv run live-translate --lang sq --translator translategemma --no-tts

# Audiodatei statt Mikrofon
uv run live-translate --lang de --file data/samples/de_test.wav

# Audiogeräte anzeigen / auswählen
uv run live-translate --list-devices
uv run live-translate --lang sq --input-device 2 --output-device 3
```

Beenden mit `Ctrl+C`. Die Tabelle zeigt pro Äußerung Quelle, Übersetzung und die Laufzeiten
(ms) für ASR, Übersetzung und TTS. Im Terminal-Modus wird kein Transkript gespeichert.

| Option | Werte | Standard |
|---|---|---|
| `--lang` | `sq`, `de` | `de` |
| `--translator` | `qwen`, `qwen-4b` (schneller, etwas schlechter), `translategemma`, `madlad` | `qwen` |
| `--context` | Anzahl vorheriger Äußerungen als Kontext für das LLM | `3` |
| `--voice` | Kokoro-Stimme (`af_heart`, `af_bella`, `am_adam`, `bf_emma`, …) | `af_heart` |
| `--no-tts` | Sprachausgabe aus | – |

## Modelle

| Stufe | Modell | RAM | Lizenz |
|---|---|---|---|
| ASR sq | `Flutra/whisper-large-v3-turbo-sq-v2` (MLX-konvertiert) | 1,6 GB | nicht angegeben |
| ASR de | `mlx-community/whisper-large-v3-turbo` | 1,6 GB | Apache 2.0 |
| MT | `mlx-community/translategemma-12b-it-4bit` | 8 GB | Gemma ToU |
| MT | `mlx-community/Qwen3.5-9B-MLX-4bit` | 6,6 GB | Apache 2.0 |
| MT (schnell, satzweise, CPU) | `Nextcloud-AI/madlad400-3b-mt-ct2-int8` | 3 GB | Apache 2.0 |
| TTS | `mlx-community/Kokoro-82M-bf16` | 0,4 GB | Apache 2.0 |
| VAD | Silero VAD v5 (ONNX) | – | MIT |

Hintergrund, Alternativen und Quellen: [RESEARCH.md](RESEARCH.md).

## Benchmark

FLORES-200 devtest, Albanisch (`als_Latn`) → Englisch:

```bash
uv run python scripts/bench_flores.py --lang sq --n 100 --translators madlad translategemma qwen
uv run python scripts/bench_flores.py --lang de --n 100 --translators translategemma qwen
```

Ergebnisse landen in `data/bench/`. Stand 15.09.2026, je 100 Sätze, M5 Pro 24 GB
(Laufzeiten teils mit parallel laufenden Prozessen gemessen, daher eher obere Grenze):

| Richtung | Modell | Kontext | chrF++ | BLEU | Ø s/Satz |
|---|---|---|---|---|---|
| sq→en | **Qwen3.5 9B 4-bit** | 3 | **65,5** | **39,6** | 1,0 |
| sq→en | Qwen3.5 9B 4-bit | 0 | 63,0 | 37,1 | 0,8 |
| sq→en | Qwen3.5 4B 4-bit (`qwen-4b`) | 3 | 61,1 | 33,0 | 0,66 (n=50) |
| sq→en | TranslateGemma 12B 4-bit | 3 | 63,0 | 34,7 | 1,6 |
| sq→en | TranslateGemma 12B 4-bit | 0 | 62,1 | 33,2 | 1,2 |
| sq→en | MADLAD-400 3B int8 (CPU) | – | 52,5 | 16,3 | 2,0 |
| de→en | Qwen3.5 9B 4-bit | 0 | 65,1 | 38,1 | 1,0 |
| de→en | TranslateGemma 12B 4-bit | 0 | 64,1 | 35,4 | 1,6 |

Fazit: Qwen3.5 9B ist Standard (beste Qualität, schnellste Antwort, Apache 2.0). Kontext aus den
vorherigen Äußerungen (`--context 3`) verbessert die Übersetzung spürbar. MADLAD lohnt sich nicht.

## Segmentierung und Robustheit

Ein Segment beginnt erst nach 0,25 s zusammenhängender Sprache und endet nach 0,4 s Stille. Ab 2 s
Länge reicht eine Pause von 0,15 s, bei 3,5 s wird an der leisesten Stelle der letzten 1,5 s geschnitten,
damit auch bei Dauerreden alle 3–4 s übersetzt wird.
Segmente mit unter 30 % Sprachanteil werden verworfen. Whisper läuft ohne Temperatur-Fallback; Segmente
mit hoher No-Speech-Wahrscheinlichkeit, bekannte Stille-Halluzinationen („Vielen Dank.“) und
Wiederholungen der letzten Äußerung werden gefiltert. Parameter in `live_translate/vad.py` und `asr.py`.
Der Übersetzer bekommt die letzten 3 Fragmente als Kontext und ist angewiesen, Fragmente als Fortsetzung
zu übersetzen, damit der Fluss-Text zusammenhängend bleibt.

**Sätze säubern:** Fragmente werden zu Blöcken gruppiert. Nach jeder endgültigen Übersetzung redigiert ein
schnelles Hilfsmodell (Qwen3.5 4B, `HELPER_TRANSLATOR` in `config.py`) inkrementell den offenen Satz plus das
neue Fragment (niedrige Priorität, abbrechbar): geteilte Sätze werden zusammengesetzt, Satzzeichen und
Großschreibung an den Schnittstellen korrigiert, Inhalt bleibt unverändert. Fertige Sätze werden nicht mehr
angefasst. Die Originalseite wird ohne LLM zusammengefügt (Satzzeichen an Schnittstellen ohne echte Pause
entfernt). Vorläufige Übersetzungen laufen ebenfalls über das Hilfsmodell, die endgültige immer über das
Hauptmodell. Ein Block schließt, wenn
das LLM den letzten Satz als vollständig meldet **und** das letzte Fragment durch eine echte Sprechpause
endete (nicht durch Zeitlimit), spätestens nach 8 Fragmenten. Die Fluss-Ansicht und das Transkript zeigen
die redigierten Blöcke, noch nicht redigierte Fragmente hängen grau hinten an. Nur mit den `qwen`-Übersetzern
verfügbar; in den Einstellungen abschaltbar.

Vorläufige Übersetzungen laufen in derselben Warteschlange wie die endgültigen, aber mit niedrigerer
Priorität: veraltete werden verworfen, und eine laufende wird tokenweise abgebrochen, sobald ein
endgültiges Segment wartet. Abschaltbar in den Einstellungen („Vorläufige Übersetzung“).

## Bekannte Grenzen

- Whisper kann Albanisch ohne Fine-Tune praktisch nicht (>50 % WER). Das Fine-Tune ist auf
  Standard-Albanisch (Tosk) trainiert; Gheg-Dialekt wird deutlich schlechter erkannt.
- Das albanische ASR-Modell ist bisher nur mit deutschem Testaudio geladen, nicht mit echtem
  albanischem Audio evaluiert.
- MADLAD übersetzt satzweise ohne Kontext und kann bei langen Eingaben halluzinieren
  (deshalb Satzsplitting + Repetition-Penalty).

## Weitergabe als App

Für Leute, die das Tool nur **benutzen** wollen. Zum Weiterentwickeln taugt die App nicht: Der Code
liegt versteckt in `~/Library/Application Support/LiveTranslate/src` und wird bei jedem Start neu
überschrieben. Dafür ist `Start.command` da (siehe Schnellstart).

`packaging/` baut ein `LiveTranslate.app`-Bundle (~21 MB), das sich beim ersten
Start selbst einrichtet — Python 3.13 und venv landen in
`~/Library/Application Support/LiveTranslate`, die Modelle im üblichen
HuggingFace-Cache `~/.cache/huggingface`. Der Empfänger sieht nie ein Terminal.

```bash
./packaging/build_app.sh                 # ad-hoc signiert -> packaging/dist/LiveTranslate.zip
```

Bestandteile:

| Datei | Zweck |
|---|---|
| `packaging/Launcher/main.swift` | AppKit-Launcher: Fortschrittsfenster, startet Setup + Server, öffnet den Browser |
| `packaging/bootstrap.sh` | idempotente Ersteinrichtung, meldet `@@STEP` / `@@PROG` / `@@READY` an den Launcher |
| `packaging/build_app.sh` | baut, bettet `uv` ein, kopiert den Quellcode als Payload, signiert, zippt |
| `scripts/prefetch.py` | lädt ein HF-Repo mit Fortschrittsmeldung |

Das Übersetzungsmodell wird aus dem RAM des Zielrechners abgeleitet
(`config.default_translator()`): ab 30 GB TranslateGemma-12B, ab 22 GB Qwen-9B,
darunter Qwen-4B. Der Bootstrap lädt genau dieses eine Modell vor.

Der albanische Whisper wird beim Setup aus `Flutra/whisper-large-v3-turbo-sq-v2`
nach MLX konvertiert. Das 3-GB-PyTorch-Original wird danach gelöscht — aber nur,
wenn es vorher nicht schon im Cache lag, damit ein bestehender Cache unangetastet
bleibt.

### Wiederholte Starts

Jeder Schritt schreibt einen Stempel nach
`~/Library/Application Support/LiveTranslate/stamps/`, und ein Stempel zählt nur,
solange das zugehörige Artefakt wirklich existiert (`verify()` in `bootstrap.sh`):

| Fall | Verhalten |
|---|---|
| Zweiter Start | alle sechs Schritte übersprungen, ~0,4 s bis zum Server |
| Modelle schon im HF-Cache | kein Download, nur Konvertierung (~20 s) |
| Abgebrochener Download | kein Stempel → Schritt läuft erneut, HF setzt den Blob fort |
| `uv.lock` geändert | `deps` läuft neu, alles andere bleibt |
| Artefakt gelöscht, Stempel da | Stempel wird verworfen, Schritt läuft erneut |

### Ohne Gatekeeper-Warnung ausliefern

Ad-hoc signiert verlangt macOS beim ersten Start einmalig
„Systemeinstellungen → Datenschutz & Sicherheit → Trotzdem öffnen". Mit einem
**Developer ID Application**-Zertifikat (im Developer-Portal anzulegen, Team
`4KAY9JCQ5X`) entfällt das:

```bash
xcrun notarytool store-credentials livetranslate --apple-id <mail> --team-id 4KAY9JCQ5X
./packaging/build_app.sh --sign "Developer ID Application: … (4KAY9JCQ5X)" --notarize livetranslate
```

Die Xcode-Beta-Sperre betrifft nur App-Store-/TestFlight-Uploads, nicht die
Developer-ID-Notarisierung.
