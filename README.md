# llm-abstention-tuning

**Teaching a small LLM to say "I don't know" instead of hallucinating — with SFT + DPO post-training, calibration and risk–coverage evaluation.**

A hallucination is a confident answer to a question the model cannot actually answer. This project post-trains
`Qwen2.5-1.5B-Instruct` so that it **answers what it knows and abstains on what it doesn't**, then measures whether
that holds up: in- and out-of-distribution, against a prompt-only baseline, and against simply thresholding the base
model's own confidence.

[![CI](https://github.com/manojrevenkar12-web/llm-abstention-tuning/actions/workflows/ci.yml/badge.svg)](https://github.com/manojrevenkar12-web/llm-abstention-tuning/actions/workflows/ci.yml)

## Results

### Run 1 — hallucinations cut 91 %, but the model over-abstains (2 of 3 blocking gates fail)

`Qwen2.5-1.5B-Instruct`, LoRA SFT + DPO on 6,000 probed TriviaQA questions, single NVIDIA L4. Full report:
[`results/qwen2.5-1.5b/REPORT.md`](results/qwen2.5-1.5b/REPORT.md).

**triviaqa (in-distribution), 1,000 questions**

| System | Accuracy | Hallucination | Abstention | Selective acc. | Over-abstention on known | ECE |
|---|---|---|---|---|---|---|
| Base | 33.2% | 66.8% | 0.0% | 33.2% | 0.0% | 0.320 |
| Base + "say I don't know" prompt | 33.0% | 66.1% | 0.9% | 33.3% | 0.3% | 0.326 |
| SFT | 18.0% | 8.1% | 73.9% | 69.0% | 41.6% | 0.126 |
| SFT + DPO | 17.1% | 5.9% | 77.0% | 74.3% | 44.7% | 0.083 |

**Out of distribution (SFT + DPO vs base)**

| Dataset | Hallucination (base → tuned) | Accuracy (base → tuned) | Abstention (tuned) |
|---|---|---|---|
| popqa | 86.4% → 2.4% | 13.2% → 5.1% | 92.5% |
| nq_open | 85.4% → 7.0% | 14.1% → 4.8% | 88.2% |

**Matched-coverage check (triviaqa):** at the same 23.0% coverage, a confidence threshold on the base model reaches 70.0% selective accuracy vs 74.3% for SFT + DPO.

![outcomes](results/qwen2.5-1.5b/outcomes.png)

**What worked.** Hallucination fell from 66.8 % to 5.9 % in-distribution and transferred out of distribution
(PopQA 86.4 % → 2.4 %, NQ-Open 85.4 % → 7.0 %). Calibration improved (ECE 0.320 → 0.083). Prompting alone does almost
nothing — the base model ignores "say I don't know" on TriviaQA (0.9 % abstention).

**What failed.** The model learned to abstain far too broadly. It refuses 44.7 % of questions the base model reliably
knows, and accuracy halves (33.2 % → 17.1 %). Against a plain confidence threshold on the base model at the same
coverage, the tuned model is only modestly better in-distribution (74.3 % vs 70.0 % selective accuracy) and level on
PopQA — most of the hallucination reduction is abstention a threshold could also buy.

**Diagnosis.**

1. *SFT data imbalance.* 63 % of SFT targets were `I don't know.` (1,577 unknown vs 943 known). Over-abstention was
   already 41.6 % after SFT, before DPO — so SFT, not DPO, set the abstain prior.
2. *DPO barely trained.* Loss moved only from 0.693 to ~0.67 and reward accuracy plateaued near 0.65
   ([training curves](results/qwen2.5-1.5b/training.png)). The learning rate (5e-6) is a full-fine-tuning DPO
   setting; LoRA adapters typically need ~10× more. DPO still helped slightly (net score +0.013 over SFT), but the
   942 *answer ≻ abstain* pairs had little chance to pull over-abstention back down.

### Run 2 — pending

[`configs/qwen2.5-1.5b-v2.yaml`](configs/qwen2.5-1.5b-v2.yaml) changes exactly those two settings — balanced SFT
data (`max_unknown_ratio: 1.0`) and DPO learning rate 5e-5 — and keeps the release gates unchanged. It reuses run 1's
probe labels. Results will be added here whether or not the gates pass.

## Method

```mermaid
flowchart LR
    A[TriviaQA questions] --> B[Probe base model<br/>greedy + K samples]
    B --> C{"p(correct)"}
    C -->|known| D[answer with its<br/>own phrasing]
    C -->|unknown| E["I don't know."]
    C -->|partial| F[DPO only]
    D --> G[LoRA SFT]
    E --> G
    G --> H[LoRA DPO<br/>correct ≻ abstain ≻ wrong]
    F --> H
    H --> I[Evaluate: TriviaQA · PopQA · NQ-Open]
    I --> J[Report + release gates]
```

1. **Probe what the model knows.** For each training question the base model gives one greedy answer and K sampled
   answers (plain prompt, no permission to abstain). The share of correct samples, p(correct), labels the question
   `known` (≥ 0.6 and greedy correct), `unknown` (never correct) or `partial`. Abstention is taught relative to *this
   model's* knowledge, not to question difficulty in general.
2. **SFT on the model's own answers.** Known → the model's own most frequent correct phrasing; unknown →
   `I don't know.` The gold string is never used as a target: fine-tuning on facts the model doesn't already hold is
   linked to *more* hallucination (Gekhman et al., 2024), so SFT only teaches **when** to answer, never **what**.
   Loss is on completion tokens only.
3. **DPO on a disjoint question set** (stratified split), with the preference order *correct ≻ abstain ≻ wrong*:
   known → `correct ≻ I don't know` (penalises over-refusal) and `correct ≻ own wrong sample`; unknown →
   `I don't know ≻ own most frequent hallucination`; partial → `correct ≻ wrong`. The reference policy is the SFT model.
4. **Evaluate with the plain prompt.** The trained models are never told they may abstain, so any abstention is in the
   weights, not the prompt.

### Baselines that the tuned model has to beat

| Baseline | Question it answers |
|---|---|
| `base` | How often does the untouched model hallucinate? |
| `base_prompted` | Can a system prompt ("if unsure, say I don't know") do the job without training? |
| **matched-coverage threshold** | If the base model abstained on its *least confident* answers exactly as often as the tuned model abstains, would it do just as well? If yes, training added nothing a confidence threshold couldn't. |
| `sft` vs `sft_dpo` | Does DPO add anything over SFT alone? |

### Metrics

| Metric | Definition |
|---|---|
| Hallucination rate | wrong answers / all questions |
| Abstention rate | "I don't know" (or equivalent) / all questions |
| Selective accuracy | correct / answered — precision of what the model does say |
| Net score | accuracy − hallucination rate (a wrong answer costs as much as a right one earns) |
| Over-abstention | refusals on questions the base model reliably knows |
| Unknown hallucination | wrong answers on questions the base model never gets right |
| ECE | expected calibration error of answer confidence (exp of mean token log-prob), answered items |
| AURC / coverage @ 90 % precision | risk–coverage curve when further abstaining on the least confident answers |

All rates carry 95 % bootstrap confidence intervals. Out-of-distribution sets: **PopQA** (long-tail entities, where a
1.5B model knows little) and **NQ-Open** (search-engine questions).

### Release gates

[`configs/gates.yaml`](configs/gates.yaml) was written *before* the first full run. Blocking gates: ≥ 30 % relative
reduction in in-distribution hallucination, ≤ 5-point accuracy drop, ≤ 15 % over-abstention. Warning gates: OOD
transfer, beating the matched-coverage threshold, beating the prompt-only baseline, DPO beating SFT.
`abstention-tuning gates` exits with code 3 on a blocking failure, the same way a CI quality gate would. Failing gates
are reported, not tuned away.

## Reproduce

**Colab (recommended):** open [`notebooks/colab_run.ipynb`](notebooks/colab_run.ipynb). Each stage is one cell and
writes to Google Drive, so a disconnect only costs the current stage. Use `configs/qwen2.5-1.5b.yaml` on an L4/A100 or
`configs/qwen2.5-1.5b-quick.yaml` on a free T4.

**Locally / any GPU box:**

```bash
pip install -e ".[train,dev]"            # or: pip install -r requirements.txt && pip install -e .
abstention-tuning all --config configs/qwen2.5-1.5b.yaml
# or stage by stage: prepare, probe, build, sft, dpo, evaluate, gates, report
abstention-tuning all --config configs/qwen2.5-1.5b.yaml --from-stage evaluate   # resume
```

Output lands in `runs/<run_name>/`: probe labels, SFT/DPO datasets, LoRA adapters and merged models, per-question
predictions, `eval/metrics.json`, and `report/REPORT.md` with figures.

**Multi-GPU:** the training stages use the Hugging Face Trainer via TRL, so `accelerate launch -m abstention_tuning
sft --config ...` runs data-parallel without code changes. The reported runs used a single GPU.

## Engineering

- **Typed config** (Pydantic) — every run writes `config.resolved.yaml` next to its results.
- **Stage-by-stage pipeline** — each stage reads the previous stage's files, so any stage re-runs on its own.
- **Tests** — unit tests for grading, abstention detection, metrics (ECE, AURC, matched coverage, bootstrap), data
  builders, probing and gates, plus an **end-to-end smoke test** that builds a tiny random Qwen2-architecture model
  offline and runs probe → SFT → DPO → evaluate → gates → report on CPU in about 30 s. CI runs all of it on every push.
- **Deterministic grading** — SQuAD-style normalisation with word-boundary alias matching; no LLM judge, so no judge
  noise or self-preference.

```text
src/abstention_tuning/
  config.py       typed run config            probe.py      knowledge probing + labels
  data.py         TriviaQA / PopQA / NQ loaders builders.py   SFT + DPO data
  grading.py      answer + abstention grading train.py      LoRA SFT / DPO with TRL
  generation.py   batched generation + conf.  evaluate.py   systems × datasets + matched baseline
  metrics.py      rates, ECE, risk–coverage   gates.py      release gates
  report.py       REPORT.md + figures         pipeline.py / cli.py
configs/  notebooks/  scripts/make_tiny_model.py  tests/
```

## Limitations

- **One small model, single GPU.** Results are for a 1.5B model; larger models know more, so their known/unknown split
  and the size of the effect will differ.
- **Short-form factual QA only.** This is closed-book trivia, not long-form generation or RAG grounding.
- **Rule-based grading under-counts correct paraphrases** missing from the alias lists. It affects every system equally,
  so comparisons stay fair, but absolute accuracy is a slight under-estimate.
- **Confidence is mean token probability**, a cheap signal. Sample consistency or a trained probe would likely
  calibrate better.
- **DPO pairs come from the base model's samples**, not re-sampled from the SFT model (slightly off-policy).
- **Knowledge labels are estimates** from K samples; with K = 8 a fact recalled 1 time in 20 can be labelled unknown.

## Related work

- Zhang et al., 2024 — *R-Tuning: Instructing Large Language Models to Say "I Don't Know"* (NAACL).
- Gekhman et al., 2024 — *Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?* (EMNLP).
- Yang et al., 2023 — *Alignment for Honesty*.
- Kadavath et al., 2022 — *Language Models (Mostly) Know What They Know*.
- Rafailov et al., 2023 — *Direct Preference Optimization*.
- Geifman & El-Yaniv, 2017 — *Selective Classification for Deep Neural Networks* (risk–coverage).

## License

MIT
