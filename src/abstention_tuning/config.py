"""Typed run configuration (Pydantic), loaded from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class DatasetSpec(BaseModel):
    """One QA source. `source` is a built-in loader name or `jsonl` with `path`."""

    name: str
    source: Literal["triviaqa", "popqa", "nq_open", "jsonl"]
    split: str = "validation"
    path: str | None = None
    limit: int | None = None

    @model_validator(mode="after")
    def _jsonl_needs_path(self) -> DatasetSpec:
        if self.source == "jsonl" and not self.path:
            raise ValueError(f"dataset '{self.name}': source=jsonl requires `path`")
        return self


class DataConfig(BaseModel):
    train: DatasetSpec
    eval: list[DatasetSpec]
    in_distribution: str = Field(description="name of the eval set that matches the training distribution")


class ModelConfig(BaseModel):
    base: str = "Qwen/Qwen2.5-1.5B-Instruct"
    precision: Literal["auto", "bf16", "fp16", "fp32"] = "auto"
    trust_remote_code: bool = False


class GenerationConfig(BaseModel):
    batch_size: int = 32
    max_new_tokens: int = 24


class ProbeConfig(BaseModel):
    num_samples: int = 8
    temperature: float = 0.7
    top_p: float = 0.95
    known_threshold: float = Field(0.6, description="p(correct) >= this -> known")
    unknown_threshold: float = Field(0.0, description="p(correct) <= this and greedy wrong -> unknown")


class BuildConfig(BaseModel):
    dpo_fraction: float = Field(0.5, ge=0.0, lt=1.0, description="share of probed questions held out for DPO")
    partial_policy_sft: Literal["skip", "answer", "abstain"] = "skip"
    max_unknown_ratio: float | None = Field(
        None, description="cap #unknown SFT examples at ratio * #known (None = natural ratio)"
    )
    known_rejects_abstain: bool = True


class LoraSpec(BaseModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] | str = "all-linear"


class SFTSpec(BaseModel):
    epochs: float = 2.0
    learning_rate: float = 1e-4
    batch_size: int = 16
    grad_accum: int = 1
    max_length: int = 256
    warmup_ratio: float = 0.05
    gradient_checkpointing: bool = False


class DPOSpec(BaseModel):
    epochs: float = 1.0
    learning_rate: float = 5e-6
    beta: float = 0.1
    batch_size: int = 8
    grad_accum: int = 2
    max_length: int = 256
    warmup_ratio: float = 0.1
    gradient_checkpointing: bool = False


class TrainConfig(BaseModel):
    lora: LoraSpec = LoraSpec()
    sft: SFTSpec = SFTSpec()
    dpo: DPOSpec = DPOSpec()


class EvalConfig(BaseModel):
    systems: list[Literal["base", "base_prompted", "sft", "sft_dpo"]] = [
        "base",
        "base_prompted",
        "sft",
        "sft_dpo",
    ]
    probe_samples: int = Field(5, description="samples per eval question to label known/unknown (0 = off)")
    ece_bins: int = 10
    bootstrap: int = 1000
    hallucination_penalty: float = 1.0
    target_precision: float = 0.9


class RunConfig(BaseModel):
    run_name: str = "qwen2.5-1.5b"
    output_dir: str = "runs"
    seed: int = 42
    model: ModelConfig = ModelConfig()
    data: DataConfig
    generation: GenerationConfig = GenerationConfig()
    probe: ProbeConfig = ProbeConfig()
    build: BuildConfig = BuildConfig()
    train: TrainConfig = TrainConfig()
    eval: EvalConfig = EvalConfig()

    @model_validator(mode="after")
    def _in_dist_exists(self) -> RunConfig:
        names = [d.name for d in self.data.eval]
        if self.data.in_distribution not in names:
            raise ValueError(f"in_distribution '{self.data.in_distribution}' not in eval sets {names}")
        if len(set(names)) != len(names):
            raise ValueError("eval dataset names must be unique")
        return self

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.run_name


def load_config(path: str | Path) -> RunConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return RunConfig.model_validate(raw)
