"""Benchmark translators on FLORES-200 devtest (source -> eng_Latn).

usage: uv run python scripts/bench_flores.py --lang sq --n 100 --translators madlad translategemma qwen
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import sacrebleu

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from live_translate.config import PROFILES, ROOT, TRANSLATORS  # noqa: E402
from live_translate.translate import make_translator  # noqa: E402

FLORES = ROOT / "data" / "flores200_dataset" / "devtest"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=PROFILES, default="sq")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--context", type=int, default=0, help="previous sentence pairs as context (LLMs)")
    ap.add_argument("--translators", nargs="+", default=list(TRANSLATORS), choices=list(TRANSLATORS))
    args = ap.parse_args()

    prof = PROFILES[args.lang]
    src = (FLORES / f"{prof.flores}.devtest").read_text().splitlines()[args.offset : args.offset + args.n]
    ref = (FLORES / "eng_Latn.devtest").read_text().splitlines()[args.offset : args.offset + args.n]
    out_dir = ROOT / "data" / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name in args.translators:
        print(f"\n=== {name} ({prof.name}→en, n={len(src)}, context={args.context}) ===", flush=True)
        tr = make_translator(name, context_turns=args.context)
        tr.translate("Hello.", prof.code, [])  # warm-up
        hyps, times, history = [], [], []
        for i, s in enumerate(src):
            h, dt = tr.translate(s, prof.code, history)
            h = h.replace("\n", " ").strip()
            hyps.append(h)
            times.append(dt)
            if args.context:
                history.append((s, h))
                history = history[-args.context :]
            if i < 3 or i % 25 == 0:
                print(f"[{i}] {dt*1000:.0f}ms  {s[:70]}\n     -> {h[:90]}", flush=True)
        chrf = sacrebleu.corpus_chrf(hyps, [ref], word_order=2)
        bleu = sacrebleu.corpus_bleu(hyps, [ref])
        r = {
            "chrF++": round(chrf.score, 2),
            "BLEU": round(bleu.score, 2),
            "mean_s": round(sum(times) / len(times), 3),
            "p90_s": round(sorted(times)[int(0.9 * len(times)) - 1], 3),
            "n": len(src),
            "context": args.context,
        }
        results[name] = r
        print(f"--> {name}: {r}", flush=True)
        (out_dir / f"{args.lang}_{name}_ctx{args.context}.hyp.txt").write_text("\n".join(hyps) + "\n")
        del tr
        import gc

        gc.collect()

    tag = f"{args.lang}_n{args.n}_ctx{args.context}"
    (out_dir / f"{tag}.json").write_text(json.dumps(results, indent=2))
    print(f"\n| model | chrF++ | BLEU | mean s | p90 s |\n|---|---|---|---|---|")
    for k, r in results.items():
        print(f"| {k} | {r['chrF++']} | {r['BLEU']} | {r['mean_s']} | {r['p90_s']} |")


if __name__ == "__main__":
    main()
