# Belgrade Group Recommender — common developer tasks
.PHONY: venv install test ingest smoke extract demo-retrieve build-vectors streamlit-ui api eval db-load docker-api-build docker-api-up

PYTHON ?= python3
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PIP) install -e ".[dev]"

test:
	$(PY) -m pytest -q

# Override paths: make ingest INGEST_INPUT=~/data/events.jsonl
INGEST_INPUT ?= data/raw/events.jsonl
INGEST_OUTPUT ?= data/processed/events_normalized.jsonl
ingest:
	$(PY) scripts/ingest_events.py --input "$(INGEST_INPUT)" --output "$(INGEST_OUTPUT)"

SMOKE_INPUT ?= data/processed/events_normalized.jsonl
SMOKE_OUTPUT ?= fixtures/events_smoke.jsonl
SMOKE_SIZE ?= 150
smoke:
	$(PY) scripts/build_smoke_dataset.py --input "$(SMOKE_INPUT)" --output "$(SMOKE_OUTPUT)" --sample-size $(SMOKE_SIZE)

STRUCTURED_INPUT ?= data/processed/events_normalized.jsonl
STRUCTURED_OUTPUT ?= data/processed/events_structured.jsonl
extract:
	$(PY) scripts/extract_event_attributes.py --input "$(STRUCTURED_INPUT)" --output "$(STRUCTURED_OUTPUT)"

demo-retrieve:
	$(PY) scripts/demo_retrieval.py --fixture-user ana --top-k 5

VECTORS_INPUT ?= data/processed/events_normalized.jsonl
VECTORS_DIR ?= data/processed/vectors
# Pass VECTORS_FLAGS= to index all rows (default: only include_in_index)
VECTORS_FLAGS ?= --only-indexable
build-vectors:
	$(PY) scripts/build_vector_index.py --input "$(VECTORS_INPUT)" --output-dir "$(VECTORS_DIR)" $(VECTORS_FLAGS)

streamlit-ui:
	$(VENV)/bin/streamlit run src/belgrade_recommender/ui/streamlit_app.py

api:
	$(VENV)/bin/uvicorn belgrade_recommender.api.main:app --reload --host 127.0.0.1 --port 8000

eval:
	$(PY) scripts/run_eval.py --spec fixtures/eval_labels.json

DB_INPUT ?= data/processed/events_normalized.jsonl
DB_OUTPUT ?= data/processed/events.sqlite
db-load:
	$(PY) scripts/load_jsonl_to_sqlite.py --input "$(DB_INPUT)" --output-db "$(DB_OUTPUT)"

# FastAPI in Docker (port 8000 → container 8080). Docker Desktop morate za ovo da instalirate
docker-api-build:
	docker compose build api

docker-api-up:
	docker compose up api
