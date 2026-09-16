from __future__ import annotations

import re
import time
from typing import Callable, Protocol

from .config import PROFILES, TRANSLATORS

History = list[tuple[str, str]]  # (source, english)

_SENT_END = re.compile(r"(?<=[.!?…])\s+(?=[\"'“„(]?[A-ZÄÖÜËÇ0-9])")
# tokens after which a period is not a sentence end (German, Albanian, English)
_ABBREV = {
    "dr", "prof", "mr", "mrs", "ms", "st", "nr", "no", "ca", "bzw", "z.b", "u.a", "d.h", "usw", "vs", "etc",
    "jr", "sr", "inc", "ltd", "co", "p.sh", "sh", "z", "b", "dhjetor", "janar",
}


def split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    for piece in _SENT_END.split(text.strip()):
        piece = piece.strip()
        if not piece:
            continue
        last = piece.rstrip(".!?…").split()[-1].lower().strip("\"'()") if piece.rstrip(".!?…").split() else ""
        # glue to previous if previous ended with an abbreviation or a single initial (e.g. "J.")
        if parts and (parts[-1].rstrip(".").split()[-1].lower().strip("\"'()") in _ABBREV or _is_initial(parts[-1])):
            parts[-1] = parts[-1] + " " + piece
        else:
            parts.append(piece)
    return parts or [text.strip()]


def _is_initial(s: str) -> bool:
    w = s.rstrip(".").split()
    return bool(w) and ((len(w[-1]) == 1 and w[-1].isalpha()) or w[-1].isdigit())


class Translator(Protocol):
    name: str

    def translate(
        self,
        text: str,
        src: str,
        history: History,
        on_partial: Callable[[str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> tuple[str, float]: ...


class LLMTranslator:
    """TranslateGemma or a general chat LLM via mlx-lm, with previous sentence pairs as context."""

    def __init__(self, repo: str, kind: str, context_turns: int = 3, max_tokens: int = 256):
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler

        self.name = repo.split("/")[-1]
        self.kind = kind
        self.context_turns = context_turns
        self.max_tokens = max_tokens
        self.model, self.tok = load(repo)
        if kind == "translategemma":
            self.tok.add_eos_token("<end_of_turn>")  # mlx-community config only lists <eos>
        self._stream = stream_generate
        self.sampler = make_sampler(temp=0.0)

    def _messages(self, text: str, src: str, history: History) -> list[dict]:
        ctx = history[-self.context_turns :] if self.context_turns else []
        if self.kind == "translategemma":
            def user(t: str) -> dict:
                return {
                    "role": "user",
                    "content": [
                        {"type": "text", "source_lang_code": src, "target_lang_code": "en", "text": t}
                    ],
                }
            msgs: list[dict] = []
            for s, e in ctx:
                msgs.append(user(s))
                msgs.append({"role": "assistant", "content": e})
            msgs.append(user(text))
            return msgs

        lang = PROFILES[src].name if src in PROFILES else src
        system = (
            f"You are a professional simultaneous interpreter at a university lecture. The user sends transcripts of "
            f"spoken {lang} as they arrive, a few seconds at a time, so an utterance is often a fragment of a longer "
            "sentence that continues in the next one. Translate each fragment into natural, fluent English so that "
            "the fragments read as one continuous text: continue the previous fragment, do not repeat what was "
            "already translated, do not complete or invent an ending, and do not add a full stop if the thought is "
            "clearly unfinished. Preserve meaning, tone and register; use the previous fragments to resolve pronouns, "
            "references and terminology. Never answer, comment or explain. Output only the English translation."
        )
        msgs = [{"role": "system", "content": system}]
        for s, e in ctx:
            msgs.append({"role": "user", "content": s})
            msgs.append({"role": "assistant", "content": e})
        msgs.append({"role": "user", "content": text})
        return msgs

    def translate(
        self,
        text: str,
        src: str,
        history: History,
        on_partial: Callable[[str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> tuple[str, float]:
        """should_stop is polled between tokens; used to abort a provisional translation early."""
        t0 = time.perf_counter()
        kwargs = {} if self.kind == "translategemma" else {"enable_thinking": False}
        prompt = self.tok.apply_chat_template(
            self._messages(text, src, history), add_generation_prompt=True, tokenize=False, **kwargs
        )
        out = ""
        for r in self._stream(self.model, self.tok, prompt=prompt, max_tokens=self.max_tokens, sampler=self.sampler):
            out += r.text
            if on_partial is not None:
                on_partial(out)
            if should_stop is not None and should_stop():
                break
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
        return out, time.perf_counter() - t0


    def complete(
        self, system: str, user: str, max_tokens: int = 400, should_stop: Callable[[], bool] | None = None
    ) -> str | None:
        """Free-form instruction call (chat models only). Returns None if aborted via should_stop."""
        if self.kind != "chat":
            return None
        prompt = self.tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            add_generation_prompt=True, tokenize=False, enable_thinking=False,
        )
        out = ""
        for r in self._stream(self.model, self.tok, prompt=prompt, max_tokens=max_tokens, sampler=self.sampler):
            out += r.text
            if should_stop is not None and should_stop():
                return None
        return re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()


class MadladTranslator:
    """MADLAD-400 3B-MT, int8 on CPU via CTranslate2. Sentence-level, no context, fast draft."""

    def __init__(self, repo: str, beam_size: int = 1, threads: int = 4):
        import ctranslate2
        import sentencepiece as spm
        from huggingface_hub import snapshot_download

        self.name = repo.split("/")[-1]
        path = snapshot_download(repo)
        self.tr = ctranslate2.Translator(path, device="cpu", compute_type="int8", intra_threads=threads)
        self.sp = spm.SentencePieceProcessor(model_file=f"{path}/spiece.model")
        self.beam_size = beam_size
        assert self.sp.piece_to_id("<2en>") != self.sp.unk_id(), "<2en> not in vocab"

    def translate(
        self,
        text: str,
        src: str,
        history: History,
        on_partial: Callable[[str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> tuple[str, float]:
        t0 = time.perf_counter()
        sents = split_sentences(text)
        batch = [["<2en>"] + self.sp.encode(s, out_type=str) for s in sents]
        res = self.tr.translate_batch(
            batch,
            beam_size=self.beam_size,
            max_decoding_length=min(256, 2 * max(len(p) for p in batch) + 10),
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )
        out = " ".join(self.sp.decode(r.hypotheses[0]).strip() for r in res)
        return out.strip(), time.perf_counter() - t0


def make_translator(name: str, context_turns: int = 3) -> Translator:
    kind, repo = TRANSLATORS[name]
    if kind == "madlad":
        return MadladTranslator(repo)
    return LLMTranslator(repo, kind, context_turns=context_turns)
