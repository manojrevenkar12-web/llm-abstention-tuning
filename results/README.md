# results/

Committed outputs of full runs, copied here by the last cell of `notebooks/colab_run.ipynb`:

```text
results/<run_name>/
  REPORT.md              generated report (tables, headline, gates)
  *.png                  outcomes, risk–coverage, reliability, training curves
  metrics.json           every metric, per dataset and system
  gates.json             release-gate results
  build_stats.json       probe labels and training-set composition
  config.resolved.yaml   exact configuration of the run
```

Models and per-question predictions are not committed (size); adapters can be pushed to the Hugging Face Hub instead.
