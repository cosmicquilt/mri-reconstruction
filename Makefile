# Single-command runner for Linux/macOS (and Docker). Windows users: use run.ps1.
export PYTHONPATH := src

.PHONY: smoke test train eval figures docker-build docker-run clean

smoke:
	python -m mri_recon.cli smoke

test:
	python scripts/smoke_test.py
	pytest -q

train:
	python -m mri_recon.cli train --config configs/synthetic.yaml

eval:
	python -m mri_recon.cli eval --config configs/eval.yaml

figures:
	python -m mri_recon.cli figures --config configs/eval.yaml

docker-build:
	docker build -t mri-recon .

docker-run:
	docker run --rm mri-recon smoke

clean:
	rm -rf results __pycache__ src/**/__pycache__ .pytest_cache *.egg-info
