"""Batched Hugging Face generation with per-answer confidence.

Confidence of a greedy answer = exp(mean token log-probability) of the generated answer
tokens (stop token excluded). It is cheap (one pass) and is what the calibration and
risk-coverage metrics rank on.
"""

from __future__ import annotations

import gc
import logging
import math
from collections.abc import Sequence

from .grading import clean_answer
from .prompts import build_messages

log = logging.getLogger(__name__)


def resolve_dtype(precision: str):
    import torch

    if precision == "fp32":
        return torch.float32
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    # auto
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _batches(seq: Sequence, size: int):
    for i in range(0, len(seq), size):
        yield i, seq[i : i + size]


class Generator:
    def __init__(self, model_path: str, precision: str = "auto", trust_remote_code: bool = False,
                 batch_size: int = 32, max_new_tokens: int = 24):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens

        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=resolve_dtype(precision), trust_remote_code=trust_remote_code
        ).to(self.device)
        self.model.eval()

        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        self.eos_ids = {e for e in [*eos, self.tok.eos_token_id] if e is not None}
        log.info("loaded %s on %s (eos ids %s)", model_path, self.device, sorted(self.eos_ids))

    # ------------------------------------------------------------------ helpers

    def _encode(self, questions: Sequence[str], abstain_instruction: bool):
        texts = [
            self.tok.apply_chat_template(build_messages(q, abstain_instruction), tokenize=False,
                                         add_generation_prompt=True)
            for q in questions
        ]
        return self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)

    def _gen_kwargs(self, do_sample: bool, temperature: float | None = None, top_p: float | None = None,
                    **extra):
        from transformers import GenerationConfig

        gc_ = GenerationConfig(
            **extra,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            top_k=None,
            eos_token_id=sorted(self.eos_ids),
            pad_token_id=self.tok.pad_token_id,
        )
        return {"generation_config": gc_}

    def _valid_mask(self, gen_tokens):
        """True for answer tokens: everything before the first stop token."""
        torch = self.torch
        is_eos = torch.zeros_like(gen_tokens, dtype=torch.bool)
        for e in self.eos_ids:
            is_eos |= gen_tokens == e
        # positions at or after the first eos are invalid
        return torch.cumsum(is_eos.int(), dim=1) == 0

    # ------------------------------------------------------------------ public API

    def greedy(self, questions: Sequence[str], abstain_instruction: bool = False) -> list[dict]:
        torch = self.torch
        out: list[dict] = []
        for _, chunk in _batches(list(questions), self.batch_size):
            enc = self._encode(chunk, abstain_instruction)
            with torch.inference_mode():
                res = self.model.generate(**enc, **self._gen_kwargs(False, output_scores=True,
                                                                    return_dict_in_generate=True))
            gen = res.sequences[:, enc["input_ids"].shape[1]:]
            logp = self.model.compute_transition_scores(res.sequences, res.scores, normalize_logits=True)
            mask = self._valid_mask(gen)
            logp = torch.where(mask, logp.float(), torch.zeros_like(logp, dtype=torch.float))
            lengths = mask.sum(dim=1)
            for row in range(gen.shape[0]):
                n_tok = int(lengths[row])
                mean_lp = float(logp[row].sum() / n_tok) if n_tok else -math.inf
                text = clean_answer(self.tok.decode(gen[row][mask[row]], skip_special_tokens=True))
                out.append({"text": text, "confidence": math.exp(mean_lp) if n_tok else 0.0,
                            "num_tokens": n_tok})
        return out

    def sample(self, questions: Sequence[str], n: int, temperature: float, top_p: float,
               abstain_instruction: bool = False) -> list[list[str]]:
        torch = self.torch
        per_batch = max(1, self.batch_size // max(1, n))
        out: list[list[str]] = []
        for _, chunk in _batches(list(questions), per_batch):
            enc = self._encode(chunk, abstain_instruction)
            with torch.inference_mode():
                seqs = self.model.generate(**enc, **self._gen_kwargs(True, temperature, top_p,
                                                                     num_return_sequences=n))
            gen = seqs[:, enc["input_ids"].shape[1]:]
            mask = self._valid_mask(gen)
            texts = [clean_answer(self.tok.decode(gen[i][mask[i]], skip_special_tokens=True))
                     for i in range(gen.shape[0])]
            out.extend(texts[j * n:(j + 1) * n] for j in range(len(chunk)))
        return out

    def close(self) -> None:
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
