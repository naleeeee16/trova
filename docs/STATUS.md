# Stanje projekta 

## Urađeno u kodu

| Oblast | Opis |
|--------|------|
| **Faza 0** | Struktura repoa (`src/belgrade_recommender/…`), `README`, `Makefile`, `.env.example`, `.gitignore`, moduli za `retrieve`, `rank`, `api`, `ui`, `schemas`, `fixtures/` |
| **Faza 1** | Ingest: JSONL → `NormalizedEvent` (`event_id`, placeholderi, tagovi, raw/resolved tekstovi) — `ingest/`, `scripts/ingest_events.py` |
| **Faza 1b** | Heuristika `include_in_index` / `exclude_reason` — `ingest/heuristics.py` |
| **Fixtures** | `fixtures/synthetic_users.json` (4 osobe, 2 grupe), `scripts/build_smoke_dataset.py`, `fixtures/events_smoke.jsonl` |
| **Gemini (deo Faze 6)** | Preferencije: `llm/gemini_preferences.py`. Događaji: `llm/gemini_event_attributes.py` + lazy u `extract/attributes.py` (`USE_LAZY_GEMINI_EVENT_ATTRIBUTES=1`, samo kad je buka `unknown`). Posle rangiranja: `llm/gemini_explain.py`; FastAPI `include_explanation` — `pip install -e ".[gemini]"` |
| **Faza 2 (MVP)** | Rule-based atributi događaja: `schemas/event_attributes.py`, `extract/rules.py`, `extract/attributes.py`, `scripts/extract_event_attributes.py` → `events_structured.jsonl` |
| **Testovi** | `pytest` (ingest, heuristika, ekstrakt, šeme, Gemini mock, cosine, embedding tekst, FAISS indeks ako je `faiss-cpu` instaliran) |
| **Retrieval (MVP)** | `sentence-transformers` + NumPy cosine: `retrieve/text.py`, `encoder.py`, `cosine_search.py`, `service.py`, `scripts/demo_retrieval.py` — `pip install -e ".[retrieve]"` |
| **Hard filter posle retrievala** | `retrieve/hard_filters.py` + `passes_all_hard_constraints`; `retrieve_semantic_then_hard_filter` / `retrieve_from_jsonl_with_hard`; grupno `retrieve_semantic_then_group_hard_filter` |
| **Grupno rangiranje (Faza 5 MVP)** | `rank/soft_score.py`, `rank/group_rank.py` (least-misery), `scripts/demo_group_rank.py` |
| **FAISS indeks (on-disk)** | `retrieve/faiss_index.py` (IndexFlatIP), `scripts/build_vector_index.py` → `data/processed/vectors/` (`events.faiss`, `event_ids.json`, `manifest.json`); `service.retrieve_top_k_faiss` + opcija `faiss_index_dir` na semantic+hard tokovima; demo flagovi `--faiss-index` |
| **SQLite (relacioni sloj)** | `storage/sqlite_store.py`, `scripts/load_jsonl_to_sqlite.py`, `make db-load`; `retrieve/service.load_normalized_events()` učitava i `.sqlite` / `.db` |
| **Streamlit (lokalni demo)** | `ui/streamlit_app.py` — retrieval jednog korisnika + grupni pipeline; `pip install -e ".[ui,retrieve]"`, `make streamlit-ui` |
| **FastAPI (REST)** | `api/main.py` + `api/paths.py` — `GET /health`, `POST /v1/recommendations/single`, `POST /v1/recommendations/group`; `pip install -e ".[api,retrieve]"`, `make api`; Docker: `Dockerfile`, `docker-compose.yml`, `make docker-api-up` |
| **Eval (offline)** | `eval/metrics.py` (P@k, R@k, MRR), `fixtures/eval_labels.json`, `scripts/run_eval.py` / `make eval` — ne zahteva SQL |

## DA DODAMO
- Postgres / Supabase (hosted); Qdrant / pgvector (metadata + ANN u jednom upitu u bazi)
- Pun LLM u produkcijskom API-ju


** u CMD-u sam ispis opisa dogadjaja skratila na 120 karaktera ali ako treba malo vise samo u scripts\demo_retrieval.py i scripts\demo_group_rank.py ostavimo duzi isecak tipa [:500]