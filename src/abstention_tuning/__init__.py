"""abstention_tuning — teach a small LLM to say "I don't know" instead of hallucinating.

Pipeline: prepare data -> probe what the base model knows -> build SFT/DPO data from the
model's own answers -> LoRA SFT -> LoRA DPO -> evaluate (hallucination, abstention,
calibration, risk-coverage, in- and out-of-distribution) -> report -> release gates.
"""

__version__ = "0.1.0"
