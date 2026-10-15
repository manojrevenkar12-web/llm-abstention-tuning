"""LoRA post-training with TRL: SFT, then DPO on top of the merged SFT model.

Both stages train LoRA adapters and then merge them, so every stage hands a plain
Hugging Face checkpoint to the next one (and to evaluation). For DPO, TRL uses the
same weights with the adapter disabled as the frozen reference policy — i.e. the
reference is exactly the SFT model, with no second copy in memory.

Distributed: these functions run unchanged under `accelerate launch` / torchrun
(multi-GPU data parallel via the HF Trainer). The reported run used a single GPU.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .config import DPOSpec, LoraSpec, RunConfig, SFTSpec
from .generation import resolve_dtype

log = logging.getLogger(__name__)


def _lora_config(spec: LoraSpec):
    from peft import LoraConfig

    return LoraConfig(r=spec.r, lora_alpha=spec.alpha, lora_dropout=spec.dropout,
                      target_modules=spec.target_modules, task_type="CAUSAL_LM")


def _precision_flags(dtype) -> dict[str, bool]:
    import torch

    if not torch.cuda.is_available():
        return {"bf16": False, "fp16": False}
    return {"bf16": dtype == torch.bfloat16, "fp16": dtype == torch.float16}


def _load(model_path: str, cfg: RunConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = resolve_dtype(cfg.model.precision)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=cfg.model.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype,
                                                 trust_remote_code=cfg.model.trust_remote_code)
    return model, tok, dtype


def _common_args(spec: SFTSpec | DPOSpec, out_dir: Path, cfg: RunConfig, dtype) -> dict[str, Any]:
    return dict(
        output_dir=str(out_dir),
        num_train_epochs=spec.epochs,
        per_device_train_batch_size=spec.batch_size,
        gradient_accumulation_steps=spec.grad_accum,
        learning_rate=spec.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=spec.warmup_ratio,  # float in [0,1) = ratio of total steps
        max_length=spec.max_length,
        gradient_checkpointing=spec.gradient_checkpointing,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=cfg.seed,
        **_precision_flags(dtype),
    )


def _merge_and_save(trainer, tok, adapter_dir: Path, merged_dir: Path) -> None:
    trainer.save_model(str(adapter_dir))
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(str(merged_dir))
    tok.save_pretrained(str(merged_dir))
    with open(adapter_dir / "log_history.json", "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=1)
    log.info("saved adapter -> %s, merged model -> %s", adapter_dir, merged_dir)


def train_sft(base_path: str, rows: list[dict], adapter_dir: Path, merged_dir: Path, cfg: RunConfig) -> None:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    if not rows:
        raise ValueError("SFT dataset is empty — check probe labels / build settings")
    ds = Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]} for r in rows])
    model, tok, dtype = _load(base_path, cfg)
    # prompt-completion data -> TRL computes loss on the completion tokens only
    args = SFTConfig(**_common_args(cfg.train.sft, adapter_dir, cfg, dtype), completion_only_loss=True)
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok,
                         peft_config=_lora_config(cfg.train.lora))
    trainer.train()
    _merge_and_save(trainer, tok, adapter_dir, merged_dir)


def train_dpo(sft_path: str, rows: list[dict], adapter_dir: Path, merged_dir: Path, cfg: RunConfig) -> None:
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    if not rows:
        raise ValueError("DPO dataset is empty — check probe labels / build settings")
    ds = Dataset.from_list([{"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
                            for r in rows])
    model, tok, dtype = _load(sft_path, cfg)
    args = DPOConfig(**_common_args(cfg.train.dpo, adapter_dir, cfg, dtype), beta=cfg.train.dpo.beta)
    trainer = DPOTrainer(model=model, ref_model=None, args=args, train_dataset=ds, processing_class=tok,
                         peft_config=_lora_config(cfg.train.lora))
    trainer.train()
    _merge_and_save(trainer, tok, adapter_dir, merged_dir)
