CONFIG ?= configs/qwen2.5-1.5b.yaml

.PHONY: install test lint smoke run report

install:
	pip install -r requirements.txt && pip install -e .

test:
	pytest -q

lint:
	ruff check src tests scripts

smoke:  ## whole pipeline on a tiny random model, CPU, ~30 s
	python scripts/make_tiny_model.py --out runs/_tiny_model
	TRITON_INTERPRET=1 python -m abstention_tuning all --config configs/smoke.yaml || true

run:
	python -m abstention_tuning all --config $(CONFIG)

report:
	python -m abstention_tuning gates --config $(CONFIG); python -m abstention_tuning report --config $(CONFIG)
