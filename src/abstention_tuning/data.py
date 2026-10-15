"""QA data loading into one common record shape, plus JSONL helpers.

Every source is mapped to `QAItem(id, question, aliases, source)`. Hugging Face loaders
are imported lazily so the metric/grading code and its tests run without `datasets`.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import DatasetSpec


@dataclass(frozen=True)
class QAItem:
    id: str
    question: str
    aliases: list[str]
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QAItem:
        return cls(id=str(d["id"]), question=d["question"], aliases=list(d["aliases"]), source=d["source"])


# --------------------------------------------------------------------------- JSONL


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# --------------------------------------------------------------------------- record mappers


def _dedupe(seq: Iterable[str]) -> list[str]:
    out, seen = [], set()
    for s in seq:
        s = str(s).strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out


def map_triviaqa(row: dict[str, Any], idx: int) -> QAItem:
    ans = row["answer"]
    aliases = _dedupe([ans.get("value", ""), *ans.get("aliases", []), *ans.get("normalized_aliases", [])])
    return QAItem(id=str(row.get("question_id", idx)), question=row["question"], aliases=aliases, source="triviaqa")


def map_popqa(row: dict[str, Any], idx: int) -> QAItem:
    answers = row["possible_answers"]
    if isinstance(answers, str):
        answers = json.loads(answers)
    return QAItem(id=str(row.get("id", idx)), question=row["question"], aliases=_dedupe(answers), source="popqa")


def map_nq_open(row: dict[str, Any], idx: int) -> QAItem:
    q = row["question"].strip()
    if not q.endswith("?"):
        q += "?"
    q = q[0].upper() + q[1:]
    return QAItem(id=f"nq-{idx}", question=q, aliases=_dedupe(row["answer"]), source="nq_open")


def map_jsonl(row: dict[str, Any], idx: int, source: str) -> QAItem:
    aliases = row.get("aliases") or row.get("answers") or [row["answer"]]
    return QAItem(id=str(row.get("id", idx)), question=row["question"], aliases=_dedupe(aliases), source=source)


_HF = {
    "triviaqa": ("mandarjoshi/trivia_qa", "rc.nocontext", map_triviaqa),
    "popqa": ("akariasai/PopQA", None, map_popqa),
    "nq_open": ("google-research-datasets/nq_open", None, map_nq_open),
}


def load_items(spec: DatasetSpec, seed: int = 42) -> list[QAItem]:
    """Load a dataset spec, shuffle deterministically, apply `limit`."""
    if spec.source == "jsonl":
        rows = read_jsonl(spec.path)  # type: ignore[arg-type]
        items = [map_jsonl(r, i, spec.name) for i, r in enumerate(rows)]
    else:
        from datasets import load_dataset  # lazy: heavy optional dependency

        repo, subset, mapper = _HF[spec.source]
        ds = load_dataset(repo, subset, split=spec.split) if subset else load_dataset(repo, split=spec.split)
        # Shuffle indices before mapping so `limit` does not require mapping the full split.
        idx = list(range(len(ds)))
        random.Random(seed).shuffle(idx)
        take = idx if spec.limit is None else idx[: spec.limit * 2]  # headroom for filtered rows
        items = [mapper(ds[i], i) for i in take]
    items = [it for it in items if it.question.strip() and it.aliases]
    if spec.source == "jsonl":
        random.Random(seed).shuffle(items)
    if spec.limit is not None:
        items = items[: spec.limit]
    return items


def disjoint(train: list[QAItem], evals: dict[str, list[QAItem]]) -> list[QAItem]:
    """Drop any training question that also appears (normalised) in an eval set."""
    from .grading import normalize

    held = {normalize(it.question) for items in evals.values() for it in items}
    return [it for it in train if normalize(it.question) not in held]
