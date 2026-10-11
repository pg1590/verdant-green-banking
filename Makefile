.PHONY: all test app api sample pipeline data mock-full features bundle check-labels calibration
PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
all:
	$(PYTHON) run_all.py
test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest -q tests
app:
	$(PYTHON) run_all.py --serve app
api:
	$(PYTHON) run_all.py --serve api
sample:     ## 200-customer sample from the same generator -> data/sample/
	$(PYTHON) -m src.data_gen.mock
pipeline:   ## P1-P4 stages in order; seeded data, models, impacts and explanations
	$(PYTHON) scripts/run_pipeline.py
pipeline-fast:
	$(PYTHON) scripts/run_pipeline.py --fast
demo:
	$(PYTHON) run_all.py --demo --serve app
data:       ## full synthetic dataset (10,000 customers) -> data/raw/ (gitignored; ~1 min)
	$(PYTHON) -m src.data_gen.generate
mock-full: data  ## alias for the full synthetic dataset
features:   ## feature store from data/raw (else data/sample) -> data/processed/features.parquet
	$(PYTHON) -m src.features.build
bundle:     ## labelled-MOCK team bundle from the feature store -> data/processed/mock_team_bundle.json
	$(PYTHON) scripts/build_mock_bundle.py
check-labels: ## are the synthetic labels learnable but not trivial?
	$(PYTHON) scripts/check_labels.py
calibration: ## generated vs official national totals -> reports/calibration_report.md
	$(PYTHON) scripts/calibration_report.py

.PHONY: ml-train ml-train-fast ml-predict ml-test ml-eval-test
# Propensity models. Inputs: data/processed/ from `make pipeline`.
ml-train:       ## grid search + per-product model selection on train CV (~2 min)
	$(PYTHON) -m src.models.train
ml-train-fast:
	$(PYTHON) -m src.models.train --fast
ml-predict:
	$(PYTHON) -m src.models.predict
ml-test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest -q tests/ml
# Held-out test set: run ONCE when models are frozen (--force re-runs are logged).
ml-eval-test:
	$(PYTHON) -m src.models.run_test
