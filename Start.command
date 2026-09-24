#!/bin/bash
# Doppelklick-Start. Beim ersten Mal wird alles eingerichtet (uv, Python 3.13,
# Bibliotheken, Sprachmodelle), danach startet die Oberfläche im Browser.
# Braucht weder Homebrew noch die Xcode Command Line Tools: uv bringt Python mit,
# und alle Bibliotheken kommen als fertige Wheels (PortAudio steckt in sounddevice,
# espeak-ng in espeakng-loader).
set -uo pipefail
cd "$(dirname "$0")" || exit 1

PORT="${LT_PORT:-8765}"
URL="http://127.0.0.1:$PORT"
B=$'\e[1m'; D=$'\e[2m'; R=$'\e[31m'; G=$'\e[32m'; X=$'\e[0m'

step() { printf '\n%s▸ %s%s\n' "$B" "$1" "$X"; }
die()  { printf '\n%s✗ %s%s\n\n' "$R" "$1" "$X"; exit 1; }

printf '\n%sLive Translate%s\n' "$B" "$X"

# ---- Voraussetzungen ---------------------------------------------------------
# hw.optional.arm64 stays 1 even when this Terminal runs under Rosetta
[ "$(sysctl -n hw.optional.arm64 2>/dev/null)" = 1 ] \
  || die "Das läuft nur auf Macs mit Apple-Chip (M1 oder neuer) - MLX gibt es für Intel nicht."
RAM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
[ "$RAM_GB" -ge 16 ] || printf '%s! Nur %s GB RAM - 16 GB sind das Minimum, es kann knapp werden.%s\n' "$R" "$RAM_GB" "$X"

# a second double-click while it is already running just opens the page again
if curl -s -o /dev/null --max-time 1 "$URL/"; then
  printf 'Läuft schon - öffne %s\n' "$URL"
  open "$URL"; exit 0
fi

# keep the Mac awake while models download and while it translates
caffeinate -i -w $$ &

# ---- uv ----------------------------------------------------------------------
UV="$(command -v uv || true)"
[ -z "$UV" ] && [ -x "$HOME/.local/bin/uv" ] && UV="$HOME/.local/bin/uv"
if [ -z "$UV" ]; then
  step "uv wird installiert (Python-Paketmanager, einmalig) …"
  # official installer: puts uv in ~/.local/bin and adds that to PATH for new terminals
  curl -LsSf https://astral.sh/uv/install.sh | sh \
    || die "uv konnte nicht installiert werden. Internetverbindung prüfen und nochmal doppelklicken."
  UV="$HOME/.local/bin/uv"
fi

# ---- Python + Bibliotheken ---------------------------------------------------
# idempotent: after the first run this only checks uv.lock and takes a fraction of a second
step "Python und Bibliotheken …"
"$UV" sync --frozen \
  || die "Die Bibliotheken konnten nicht installiert werden (siehe oben). Nochmal doppelklicken setzt fort."

# ---- Modelle -----------------------------------------------------------------
step "Sprachmodelle …"
.venv/bin/python scripts/setup_models.py \
  || die "Ein Modell konnte nicht geladen werden (siehe oben). Nochmal doppelklicken setzt fort."

# ---- Server ------------------------------------------------------------------
step "Oberfläche startet …"
# the venv entry point directly, not `uv run`: closing this window must reach the server itself
.venv/bin/live-translate-ui --no-browser --port "$PORT" &
SERVER=$!
trap 'kill "$SERVER" 2>/dev/null' EXIT INT TERM HUP
for _ in $(seq 1 120); do
  curl -s -o /dev/null --max-time 1 "$URL/" && break
  kill -0 "$SERVER" 2>/dev/null || die "Der Server ist abgestürzt (siehe oben)."
  sleep 0.5
done
open "$URL"
printf '\n%s● Läuft:%s %s\n' "$G" "$X" "$URL"
printf '%s  Dieses Fenster offen lassen. Beenden: Fenster schließen oder Ctrl+C.%s\n' "$D" "$X"
printf '%s  Beim ersten Start fragt macOS nach dem Mikrofon - für Terminal erlauben.%s\n\n' "$D" "$X"
wait "$SERVER"
