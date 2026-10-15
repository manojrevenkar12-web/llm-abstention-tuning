"""End-to-end run on a tiny random model (CPU, ~30 s). Proves every stage hands off correctly.

Skipped automatically when the training extras are not installed.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("peft")

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_full_pipeline_on_tiny_model(tmp_path):
    model_dir = tmp_path / "tiny"
    subprocess.run([sys.executable, str(ROOT / "scripts/make_tiny_model.py"), "--out", str(model_dir),
                    "--fixtures", str(ROOT / "tests/fixtures")], check=True, cwd=ROOT)

    cfg = yaml.safe_load((ROOT / "configs/smoke.yaml").read_text())
    cfg["model"]["base"] = str(model_dir)
    cfg["output_dir"] = str(tmp_path / "runs")
    for spec in [cfg["data"]["train"], *cfg["data"]["eval"]]:
        spec["path"] = str(ROOT / spec["path"])
    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    env = dict(os.environ)
    import torch

    if not torch.cuda.is_available():
        env["TRITON_INTERPRET"] = "1"  # TRL's fused LM-head kernel is Triton; interpret it on CPU
    proc = subprocess.run([sys.executable, "-m", "abstention_tuning", "all", "--config", str(cfg_path),
                           "--gates", str(ROOT / "configs/gates.yaml")], cwd=ROOT, env=env,
                          capture_output=True, text=True)
    # exit code 3 = a gate failed, which is expected for a random model; anything else is a crash
    assert proc.returncode in (0, 3), f"pipeline crashed (exit {proc.returncode}):\n{proc.stderr[-6000:]}"

    run = tmp_path / "runs" / "smoke"
    for rel in ["data/train.jsonl", "probe/train.jsonl", "datasets/sft.jsonl", "datasets/dpo.jsonl",
                "models/sft_merged/config.json", "models/dpo_merged/config.json", "eval/metrics.json",
                "report/gates.json", "report/REPORT.md", "report/risk_coverage.png"]:
        assert (run / rel).exists(), rel

    metrics = json.loads((run / "eval/metrics.json").read_text())
    assert set(metrics) == {"fixture_eval", "fixture_ood"}
    assert set(metrics["fixture_eval"]) == {"base", "base_prompted", "sft", "sft_dpo"}
    for ds in metrics.values():
        for m in ds.values():
            assert abs(m["accuracy"] + m["hallucination_rate"] + m["abstention_rate"] - 1) < 1e-9
