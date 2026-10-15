"""Build a tiny, randomly initialised Qwen2-architecture model + tokenizer, fully offline.

Used by the smoke test and CI to run the *whole* pipeline (probe -> build -> SFT -> DPO ->
evaluate -> gates -> report) in about a minute on CPU. Its answers are noise; the point
is to prove the code paths, shapes, masking and file hand-offs work before spending GPU
time on the real model.

    python scripts/make_tiny_model.py --out runs/_tiny_model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def corpus(fixture_dir: Path) -> list[str]:
    from abstention_tuning.prompts import ABSTAIN_SYSTEM, ABSTAIN_TEXT, PLAIN_SYSTEM

    texts = [PLAIN_SYSTEM, ABSTAIN_SYSTEM, ABSTAIN_TEXT, "system user assistant"]
    for p in sorted(fixture_dir.glob("*.jsonl")):
        for line in p.read_text().splitlines():
            row = json.loads(line)
            texts.append(row["question"])
            texts.extend(row["aliases"])
    return texts


def build(out: Path, fixture_dir: Path, seed: int = 0) -> Path:
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import GenerationConfig, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=600, special_tokens=SPECIALS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(corpus(fixture_dir), trainer=trainer)

    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                  additional_special_tokens=["<|im_start|>"])
    tok.chat_template = CHAT_TEMPLATE

    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      tie_word_embeddings=True, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                      bos_token_id=None)
    model = Qwen2ForCausalLM(cfg)
    model.generation_config = GenerationConfig(eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/_tiny_model")
    ap.add_argument("--fixtures", default="tests/fixtures")
    a = ap.parse_args()
    print(build(Path(a.out), Path(a.fixtures)))
