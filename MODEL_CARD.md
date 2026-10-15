# Model card — Qwen2.5-1.5B-Instruct + abstention LoRA (SFT + DPO)

> Fill the bracketed fields from `results/<run>/REPORT.md` after a full run. Do not round in your favour.

## Summary

LoRA adapters (SFT, then DPO) on top of `Qwen/Qwen2.5-1.5B-Instruct` that make the model reply `I don't know.` to
short factual questions it cannot reliably answer, instead of guessing.

## Intended use

- Research and demonstration of hallucination reduction through post-training.
- Short-form, closed-book factual QA in English.

## Out of scope

- Long-form generation, RAG grounding, multi-turn chat, non-English text — not evaluated.
- Any high-stakes decision. A lower hallucination rate is not zero; the model still answers some questions wrongly.

## Training

| | |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Data | TriviaQA (rc.nocontext, train split), [N] questions probed |
| Knowledge labels | greedy + [K] samples per question; known / partial / unknown = [x] / [y] / [z] |
| SFT | LoRA r=16, α=32, all linear layers; [n] examples; 2 epochs; lr 1e-4 |
| DPO | LoRA r=16 on the merged SFT model; [n] pairs; β=0.1; lr 5e-6; 1 epoch |
| Hardware | [GPU type], single GPU, [hours] |

## Evaluation (plain prompt, greedy decoding)

| Dataset | System | Accuracy | Hallucination | Abstention | Selective acc. |
|---|---|---|---|---|---|
| TriviaQA (in-dist.) | Base | [ ] | [ ] | [ ] | [ ] |
| TriviaQA (in-dist.) | SFT + DPO | [ ] | [ ] | [ ] | [ ] |
| PopQA (OOD) | Base | [ ] | [ ] | [ ] | [ ] |
| PopQA (OOD) | SFT + DPO | [ ] | [ ] | [ ] | [ ] |
| NQ-Open (OOD) | Base | [ ] | [ ] | [ ] | [ ] |
| NQ-Open (OOD) | SFT + DPO | [ ] | [ ] | [ ] | [ ] |

Over-abstention on questions the base model knows: [ ]. Release gates: [pass/fail per gate].

## Known limitations

See the README's *Limitations* section. In particular: rule-based grading under-counts paraphrases, knowledge labels
are sampled estimates, and results are for a single 1.5B model.
