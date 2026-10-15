"""Prompt templates.

Two system prompts are used deliberately:

* PLAIN_SYSTEM — asks for a short answer and says nothing about abstaining. The base
  model, the probe and both trained models use it, so any abstention a trained model
  shows has been learned into the weights, not instructed by the prompt.
* ABSTAIN_SYSTEM — additionally tells the model it may say "I don't know". Used only
  for the `base_prompted` baseline: "can a prompt alone do this?"
"""

from __future__ import annotations

ABSTAIN_TEXT = "I don't know."

PLAIN_SYSTEM = (
    "Answer the question with a short answer of a few words. "
    "Reply with the answer only, no explanation."
)

ABSTAIN_SYSTEM = (
    "Answer the question with a short answer of a few words. "
    "Reply with the answer only, no explanation. "
    f"If you are not confident that you know the correct answer, reply exactly: {ABSTAIN_TEXT}"
)


def build_messages(question: str, abstain_instruction: bool = False) -> list[dict[str, str]]:
    system = ABSTAIN_SYSTEM if abstain_instruction else PLAIN_SYSTEM
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question.strip()},
    ]
