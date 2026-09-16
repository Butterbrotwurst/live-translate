# Live-Translate (Albanisch → Englisch) — Modell-Recherche

Stand: 15.09.2026 · Zielgerät: MacBook Pro M5 Pro, 24 GB Unified Memory, macOS 27
Ziel: kostenlose, komplett lokale Live-Übersetzung Albanisch→Englisch (Test-Modus Deutsch→Englisch),
mit optionaler Sprachausgabe auf Kopfhörer.

## 0. Kurzfazit

| Stufe | Empfehlung | HF-ID | RAM | Lizenz |
|---|---|---|---|---|
| ASR (sq) | Whisper large-v3-turbo, albanisches Fine-Tune | `Flutra/whisper-large-v3-turbo-sq-v2` | ~1,6 GB fp16 | ⚠️ nicht angegeben (Basis Apache 2.0) |
| ASR (sq) Genauigkeits-Referenz | Meta Omnilingual ASR CTC 1B/3B (`als_Latn`) | `facebook/omniASR-CTC-1B` (MLX-Port via speech-swift) | 0,5–2 GB | Apache 2.0 |
| ASR (de, Test) | Whisper large-v3-turbo Basis | `openai/whisper-large-v3-turbo` (`mlx-community/whisper-large-v3-turbo`) | ~1,6 GB | Apache 2.0 |
| Übersetzung Hauptpfad | TranslateGemma 12B, 4-bit | `mlx-community/translategemma-12b-it-4bit` | ~8 GB | Gemma ToU |
| Übersetzung Alternative | Qwen3.5 9B, 4-bit (bereits in Ollama) | `mlx-community/Qwen3.5-9B-MLX-4bit` / `qwen3.5:9b` | ~6,6 GB | Apache 2.0 |
| Übersetzung Schnell-Draft | MADLAD-400 3B-MT (GGUF Q8) | `google/madlad400-3b-mt`, GGUF `cstr/madlad400-3b-mt-GGUF` | ~3 GB | Apache 2.0 |
| TTS | Kokoro-82M (MLX) | `mlx-community/Kokoro-82M-bf16` | <0,5 GB | Apache 2.0 |
| TTS Streaming-Alternative | Kyutai Pocket TTS (CPU) | `kyutai/pocket-tts` | <1 GB | CC-BY-4.0 |

Speicherbudget: LLM 8 GB + ASR 1,6 GB + TTS 0,5 GB + macOS ~4 GB ≈ 14–16 GB → passt bequem in 24 GB.

---

## 1. ASR — Albanisch ist das schwächste Glied

### Kernbefund
- Außer Whisper und Metas Omnilingual ASR kann **kein** großes Standard-Modell Albanisch:
  Parakeet/Canary (25 EU-Sprachen ohne sq), Voxtral (13 Spr.), Qwen3-ASR Basis (30 Spr.), SeamlessM4T v2 — alle ohne `sq`.
- **Whisper zero-shot ist für Albanisch unbrauchbar**: nur 5,4 h Albanisch im Training (Whisper-Paper Fig. 11, vs. 13.344 h Deutsch).
  Studie 2026 (Kryeziu & Shehu): Whisper Turbo **52,9 % WER** (formal), **63,5 %** (Dialekt). → Fine-Tune ist Pflicht.
- FLEURS enthält kein Albanisch → einzige öffentliche Referenz ist Common Voice `sq` (~9 h).
- Dialekt-Bias: Modelle sind auf Standard-Albanisch (Tosk) geprägt; Gheg (Kosovo, Nordalbanien) deutlich schlechter.

### Kandidaten
| # | Modell | Größe | Albanisch-Qualität | Lizenz | Runtime Mac | Streaming |
|---|---|---|---|---|---|---|
| 1 | `Flutra/whisper-large-v3-turbo-sq-v2` | 809M | **WER 6,98 %** auf CV19-Holdout (in-domain, vorgelesen → real schlechter) | ⚠️ keine angegeben | mlx-whisper, whisper.cpp, WhisperKit, faster-whisper | ✅ WhisperLiveKit / whisper-stream |
| 2 | Omnilingual ASR `facebook/omniASR-CTC-{300M,1B,3B,7B}` / `omniASR-LLM-*` | 0,3–7B | 7B-LLM: **CER 4,6** (`als_Latn`); Gheg `aln_Latn` CER 33; Deutsch CER 1,5. CTC-Zahlen nicht veröffentlicht | Apache 2.0 | speech-swift (MLX/CoreML), sherpa-onnx | ❌ batch ≤40 s → VAD-Chunking |
| 3 | `Kushtrim/Qwen3-ASR-0.6B-Albanian(-728h)` | 0,78B | keine WER veröffentlicht | Apache 2.0, **gated** | mlx-audio (Konvertierung ungeklärt) | nur Token-Streaming |
| – | `rishabhjain16/whisper_l3_to_kaggle_sq` / `_cv_sq` | 1,54B | WER 23,8 % / 29,7 % | Apache 2.0 | wie Whisper | ✅ |
| – | `kadriu/shqip-mms-3` (MMS-1b-all) | 1B | WER 34,5 % | CC-BY-NC | transformers | ❌ |

Whisper-Varianten: large-v3 > turbo zero-shot bei Low-Resource, aber mit Fine-Tune ist turbo (4 Decoder-Layer, halbe Größe) die bessere Echtzeit-Wahl. distil-whisper: nur Englisch.

### Gegenmaßnahmen gegen Whisper-Halluzinationen
`language="sq"` immer erzwingen · Silero VAD davor · `condition_on_previous_text=False` · `compression_ratio_threshold` / `no_speech_threshold` setzen · Beam Search.

### Run-Hinweise
```bash
# A) Alles in einem, echtes Streaming (Apache 2.0)
pip install "whisperlivekit[mlx-whisper]"
wlk --backend mlx-whisper --model-path Flutra/whisper-large-v3-turbo-sq-v2 --language sq
# Deutsch-Test: --model-path openai/whisper-large-v3-turbo --language de

# B) mlx-whisper direkt (HF → MLX konvertieren)
git clone https://github.com/ml-explore/mlx-examples && cd mlx-examples/whisper
python convert.py --torch-name-or-path Flutra/whisper-large-v3-turbo-sq-v2 --mlx-path mlx_models/sq-turbo --dtype float16

# C) whisper.cpp (ist bereits per brew installiert)
python3 models/convert-h5-to-ggml.py ./whisper-large-v3-turbo-sq-v2/ ./whisper .
whisper-stream -m ggml-model.bin -l sq --step 500 --length 5000
```

---

## 2. Übersetzung SQ→EN mit Kontext

### Albanisch-Unterstützung
| Modell | Albanisch? | Evidenz |
|---|---|---|
| NLLB-200 (600M/1.3B/3.3B) | ✅ (`als_Latn`) | metrics.csv |
| MADLAD-400 3B/7B/10B-MT | ✅ (`sq`) | Paper Tab. 17/18 |
| TranslateGemma 4B/12B/27B | ⚠️ nur **en→sq** im SFT, sq→en nicht evaluiert | Tech Report Tab. 6 / App. B |
| Qwen3 / Qwen3.5 | ✅ „Tosk Albanian" explizit in der 119-Sprachen-Liste | Qwen3-Blog |
| Gemma 3 / Gemma 4 | ? „140+ Sprachen", keine Liste | unverifiziert |
| SeamlessM4T v2, Hunyuan-MT, Seed-X, Tower, X-ALMA, Aya Expanse, EuroLLM, Llama 3/4 | ❌ | Model Cards |

### Belegte sq→en-Qualität (FLORES-200)
| Modell | sq→en | Lizenz |
|---|---|---|
| NLLB-200-3.3B | chrF++ 65,9 | CC-BY-NC |
| NLLB-200-distilled-1.3B | chrF++ 64,8 | CC-BY-NC |
| NLLB-200-distilled-600M | chrF++ 62,7 | CC-BY-NC |
| MADLAD-400 MT-3B | BLEU 41,6 / chrF 66,9 | Apache 2.0 |
| MADLAD-400 MT-7B | BLEU 43,7 / chrF 68,2 | Apache 2.0 |
| opus-mt-sq-en | BLEU 58,4 (Tatoeba, leichte Sätze) | Apache 2.0 |

Für LLMs (TranslateGemma, Qwen3.5, Gemma 4) gibt es **keine** veröffentlichten sq→en-Zahlen → eigener Test nötig (FLORES-200 devtest `sqi_Latn→eng_Latn`, 50–100 Sätze, chrF/COMET).

### Steckbriefe
**TranslateGemma 12B** (`google/translategemma-12b-it`, Jan 2026): Gemma-3-Basis, SFT + RL; schlägt Gemma-3-27B auf WMT24++ (MetricX 3,60 vs 4,04). Chat-Template mit
`{"type":"text","source_lang_code":"sq","target_lang_code":"en","text":...}`, `do_sample=False`. Ollama `translategemma:12b` (8,1 GB), MLX `mlx-community/translategemma-12b-it-4bit`. ~31 tok/s auf M4 Pro → 25-Token-Satz ≈ 0,8–1 s. Für de→en identisch nutzbar (en→de MetricX 1,36).

**Qwen3.5 9B**: einziges LLM mit expliziter Albanisch-Angabe; Apache 2.0; ~40 tok/s MLX. Thinking aus (`enable_thinking=False`), System-Prompt „Translate Albanian to English. Output only the translation." Bereits lokal als `qwen3.5:9b` in Ollama vorhanden.

**MADLAD-400 3B-MT**: T5 Encoder-Decoder, Prompt `<2en> ` + Text, Satzebene, GGUF Q4 ≈2 GB / Q8 ≈3 GB, llama.cpp. Community meldet gelegentliche Wiederholungen bei langen Inputs → Satzsegmentierung vorschalten.

**Wichtig (Ollama):** Ollama nutzt MLX erst ab 32 GB Unified Memory; auf 24 GB läuft es über llama.cpp/Metal. Für MLX-Speed direkt `mlx-lm` (Python) verwenden.

### Community-Fine-Tunes (alle ohne Benchmarks — nicht als Primärmodell)
`Kushtrim/gemma4-e4b-sq-en-v2` (gated), `Kushtrim/llama-3.2-3b-instruct-sq-en`, `olsi8/gemma-3-4b-it-shqip-v3`, `kleidialogo/bleta-8B-v0.5`, `nisten/shqiponja-*` (Autor: „don't use in production").

---

## 3. TTS — realistisch, lokal, Echtzeit

Budget für TTS: ≤ 3–4 GB neben LLM + ASR. Das schließt alle ≥3B-Modelle aus (Breeze TTS 2, Fish S2 Pro, Voxtral TTS, Higgs v3, Orpheus bf16).

| # | Modell | HF-ID | Größe | Qualität | Latenz / Streaming | Mac | Lizenz |
|---|---|---|---|---|---|---|---|
| 1 | **Kokoro-82M** | `mlx-community/Kokoro-82M-bf16` (`hexgrad/Kokoro-82M`) | 82M, ~330 MB | Elo 1061 (AA open-weights #6, bestes das passt), UTMOS 4,46 | pro Satz/Clause; RTF ≈0,02–0,05 auf MLX, ~80–200 ms TTFA | mlx-audio, kokoro-mlx, CoreML | Apache 2.0 (espeak-ng GPL nötig) |
| 2 | **Pocket TTS** (Kyutai) | `kyutai/pocket-tts` | 100M | UTMOS 4,10, „clean" | **echtes Streaming**, ~200 ms TTFA, 6× RT auf CPU | CPU (PyTorch), `pocket-tts-mlx`, CoreML | CC-BY-4.0 |
| 3 | Qwen3-TTS 0.6B | `mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit` | 0,6B, ~1,5 GB | reichere Prosodie als Kokoro | Streaming in mlx-audio; Mac-RTF unveröffentlicht | mlx-audio | Apache 2.0 |
| – | Kyutai TTS 1.6B | `kyutai/tts-1.6b-en_fr` | 1,8B | gut | nimmt **Text-Stream** an (passt zu LLM-Tokens), aber 1,5 s TTFA auf M4 Pro | moshi-mlx | CC-BY-4.0 |
| – | Chatterbox Turbo | `ResembleAI/chatterbox-turbo` | 350M | sehr gut | MLX-Pfad ~17 s/Äußerung, CoreML ~1–1,5 s; braucht Referenz-Audio | CoreML | MIT |
| ✗ | Supertonic 3 | archiviert 09.09.2026, OpenRAIL-M | | | | | |
| ✗ | Soprano, Marvis, Orpheus, CSM, Dia, F5, XTTS, Piper | Robustheit / Speicher / Lizenz / Qualität | | | | | |

Kein offenes ElevenLabs- oder Gemma-TTS existiert (Stand 09/2026).

```bash
pip install mlx-audio && brew install espeak-ng
mlx_audio.tts.generate --model mlx-community/Kokoro-82M-bf16 --text "Hello." --voice af_heart --stream
# Alternative CPU-Streaming: pip install pocket-tts   (Voices: alba, marius, javert, jean, ...)
```

Praxis: Kokoro auf GPU (MLX); bei Contention mit dem LLM auf Pocket TTS (CPU) wechseln — dann stehen ASR (GPU), LLM (GPU) und TTS (CPU) nicht in einer Queue.

---

## 4. Vorgeschlagene Architektur

```
Mikro (portaudio) → Silero VAD (bereits im HF-Cache, MLX)
  → ASR: mlx-whisper mit sq-Fine-Tune  [Test: whisper-large-v3-turbo, language=de]
  → Satzsegmentierung
  → Übersetzung: MADLAD-3B Draft (sofort anzeigen)  →  LLM-Refine mit 2–3 Sätzen Kontext (TranslateGemma 12B oder Qwen3.5 9B)
  → TTS: Kokoro-82M (MLX) → Kopfhörer
UI: Web-Frontend (WebSocket) oder SwiftUI-Menüleisten-App
```
Stack: Python 3.13 + uv, mlx-whisper, mlx-lm, mlx-audio, llama.cpp (MADLAD). Deutsch-Test = nur ASR-Modell + `language` tauschen, Übersetzer identisch.

## 5. Offene Punkte / nächste Schritte
1. FLORES-Benchmark sq→en: TranslateGemma 12B vs Qwen3.5 9B vs MADLAD 3B (~30 Min).
2. Lizenz von `Flutra/whisper-large-v3-turbo-sq-v2` klären (Issue/Discussion auf HF) — für privaten Gebrauch unkritisch.
3. Fine-Tune auf echtem Albanisch-Audio (Vorlesung, Gespräch) testen — 6,98 % ist in-domain Common Voice.
4. Zugang zu `Kushtrim/Qwen3-ASR-0.6B-Albanian-728h` anfragen.
5. Mac-RTF von Qwen3-TTS 0.6B messen, falls Kokoro klanglich nicht reicht.

## Quellen (Auswahl)
- Whisper-Paper: https://arxiv.org/pdf/2212.04356 · Albanisch-Studie 2026: https://online-journals.org/index.php/i-joe/article/view/58873
- https://huggingface.co/Flutra/whisper-large-v3-turbo-sq-v2 · https://huggingface.co/Kushtrim/Qwen3-ASR-0.6B-Albanian-728h
- Omnilingual ASR: https://huggingface.co/facebook/omniASR-LLM-7B · https://github.com/facebookresearch/omnilingual-asr/blob/main/per_language_results_table_7B_llm_asr.csv · https://github.com/soniqo/speech-swift/blob/main/docs/models/omnilingual-asr.md
- WhisperLiveKit: https://github.com/QuentinFuxa/WhisperLiveKit · mlx-whisper: https://github.com/ml-explore/mlx-examples/tree/main/whisper
- TranslateGemma Report: https://arxiv.org/pdf/2601.09012 · https://huggingface.co/google/translategemma-12b-it
- MADLAD-400: https://arxiv.org/pdf/2309.04662 · https://huggingface.co/google/madlad400-3b-mt
- NLLB metrics: https://dl.fbaipublicfiles.com/large_objects/nllb/models/nllb_200_dense_3b/metrics.csv
- Qwen3 Sprachliste: https://qwenlm.github.io/blog/qwen3/ · https://huggingface.co/Qwen/Qwen3.5-9B
- TTS-Leaderboard: https://artificialanalysis.ai/text-to-speech/leaderboard/open-weights · CPU-Benchmark: https://heyneo.com/blog/kokoro-supertonic-inflect-nano-pocket-tts-cpu-benchmark
- Kokoro: https://huggingface.co/hexgrad/Kokoro-82M · mlx-audio: https://github.com/Blaizzy/mlx-audio · Pocket TTS: https://github.com/kyutai-labs/pocket-tts
