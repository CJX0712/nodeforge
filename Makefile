PY ?= python

setup:            ## create venv + install deps (torch CPU first)
	$(PY) -m venv .venv
	.venv/Scripts/python -m pip install -U pip
	-.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	.venv/Scripts/python -m pip install -r requirements.txt

lint:
	ruff check nodeforge tests examples

test:
	pytest -q -W ignore::UserWarning

cov:
	pytest -q -W ignore::UserWarning --cov=nodeforge --cov-report=term

demo:
	python examples/run_demo.py

benchmark:
	python -m nodeforge.cli benchmark --seeds 5

.PHONY: setup lint test cov demo benchmark
