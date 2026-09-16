#!/bin/bash
# First-run setup for LiveTranslate.app. Idempotent: every step writes a stamp
# and is skipped on later launches. Driven by the Swift launcher, which reads
#   @@STEP <n>/<total> <text>   and   @@PROG <0.0-1.0>
# from stdout. Everything else is log output.
set -uo pipefail

RES="${1:?resources dir missing}"
APPSUP="$HOME/Library/Application Support/LiveTranslate"
SRC="$APPSUP/src"
STAMPS="$APPSUP/stamps"
TOTAL=6

# standard cache location, so models another tool already downloaded are reused
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export UV_CACHE_DIR="$APPSUP/uv-cache"
export UV_PYTHON_INSTALL_DIR="$APPSUP/python"
export UV_NO_CONFIG=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
UV="$RES/bin/uv"

mkdir -p "$STAMPS" "$HF_HOME"

step() { printf '@@STEP %s/%s %s\n' "$1" "$TOTAL" "$2"; }
prog() { printf '@@PROG %s\n' "$1"; }
fail() { printf '@@FAIL %s\n' "$1"; exit 1; }
done_step() { touch "$STAMPS/$1"; prog 1.0; }
have() { [ -f "$STAMPS/$1" ]; }
# a stamp only counts if the thing it stands for still exists
verify() { [ -f "$STAMPS/$1" ] && [ -e "$2" ] || { rm -f "$STAMPS/$1"; return 1; }; }
hf_dir() { printf '%s/hub/models--%s' "$HF_HOME" "${1//\//--}"; }

# 1 -------------------------------------------------------------- payload ---
step 1 "Programmdateien werden kopiert …"
mkdir -p "$SRC"
rsync -a --delete \
  --exclude '.venv' --exclude '__pycache__' --exclude 'models' \
  --exclude 'transcripts' --exclude 'data/flores200_dataset' \
  "$RES/payload/" "$SRC/" || fail "Programmdateien konnten nicht kopiert werden."
done_step payload

# 2 -------------------------------------------------------------- python ----
step 2 "Python wird eingerichtet …"
if ! verify python "$UV_PYTHON_INSTALL_DIR"; then
  "$UV" python install 3.13 || fail "Python 3.13 konnte nicht installiert werden."
  done_step python
else
  prog 1.0
fi

# 3 ---------------------------------------------------------------- deps ----
step 3 "Bibliotheken werden installiert (das dauert einige Minuten) …"
if ! verify deps "$SRC/.venv/bin/live-translate-ui" || [ "$SRC/uv.lock" -nt "$STAMPS/deps" ]; then
  "$UV" sync --directory "$SRC" --frozen || fail "Die Bibliotheken konnten nicht installiert werden."
  done_step deps
else
  prog 1.0
fi

PY=("$UV" run --directory "$SRC" --frozen python)

# 4 ------------------------------------------------------------ translator --
# the model is picked from this Mac's RAM, so a 16 GB machine gets the 4B one
MT_REPO="$("${PY[@]}" -c 'from live_translate.config import TRANSLATORS, default_translator; print(TRANSLATORS[default_translator()][1])')"
step 4 "Übersetzungsmodell wird geladen ($MT_REPO) …"
if ! verify mt "$(hf_dir "$MT_REPO")"; then
  "${PY[@]}" "$SRC/scripts/prefetch.py" "$MT_REPO" \
    || fail "Das Übersetzungsmodell konnte nicht geladen werden."
  done_step mt
else
  prog 1.0
fi

# 5 ------------------------------------------------------------------ tts ---
step 5 "Sprachausgabe wird geladen (ca. 0,4 GB) …"
if ! verify tts "$(hf_dir mlx-community/Kokoro-82M-bf16)"; then
  "${PY[@]}" "$SRC/scripts/prefetch.py" mlx-community/Kokoro-82M-bf16 \
    || fail "Die Sprachausgabe konnte nicht geladen werden."
  done_step tts
else
  prog 1.0
fi

# 6 ------------------------------------------------------------------ asr ---
step 6 "Spracherkennung Albanisch wird geladen (ca. 3 GB) …"
MLX_SQ="$SRC/models/whisper-large-v3-turbo-sq-v2-mlx"
if ! verify asr "$MLX_SQ/weights.safetensors"; then
  SQ_CACHE="$(hf_dir Flutra/whisper-large-v3-turbo-sq-v2)"
  SQ_WAS_CACHED=no; [ -d "$SQ_CACHE" ] && SQ_WAS_CACHED=yes
  "${PY[@]}" "$SRC/scripts/prefetch.py" Flutra/whisper-large-v3-turbo-sq-v2 \
    --allow '*.json' 'model.safetensors' '*.txt' \
    || fail "Die Spracherkennung konnte nicht geladen werden."
  printf 'Modell wird für Apple Silicon umgewandelt …\n'
  prog 0.9
  "${PY[@]}" "$SRC/scripts/convert_whisper.py" \
    --torch-name-or-path Flutra/whisper-large-v3-turbo-sq-v2 \
    --mlx-path "$MLX_SQ" --dtype float16 \
    || fail "Die Spracherkennung konnte nicht umgewandelt werden."
  # the 3 GB PyTorch original is no longer needed once converted -- but only
  # throw it away if it was not already in the cache before we started
  if [ "$SQ_WAS_CACHED" = no ]; then
    printf 'PyTorch-Original wird entfernt (%s)\n' "$SQ_CACHE"
    rm -rf "$SQ_CACHE"
  fi
  done_step asr
else
  prog 1.0
fi

printf '@@READY\n'
