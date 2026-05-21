# Code Guide — Belgrade Group Event Recommender

A walkthrough of every source file: what it does, what data it works with, and how it fits into the overall flow.

---

## How the whole system works (the pipeline)

```
Raw JSONL file
     │
     ▼
[1. INGEST]  Parse → clean → tag → heuristic filter → NormalizedEvent
     │
     ▼
[2. EXTRACT]  Rule-based text analysis → EventAttributes (price, city, noise, type)
     │
     ▼
[3. RETRIEVE]  Embed events + user query → cosine/FAISS search → top-K candidates
                    └─ optional hard-filter pass (city, price, banned categories)
     │
     ▼
[4. RANK]  Per-user soft scoring → least-misery group aggregation
     │
     ▼
[5. OUTPUT]  Streamlit UI  /  FastAPI REST  /  CLI demo scripts
                    └─ optional Gemini explanation step
```

Each step is its own Python module. Data flows as Pydantic models between them.

---

## Key data types (the "lingua franca")

| Type | Where defined | What it holds |
|------|--------------|---------------|
| `RawEventRecord` | `ingest/models.py` | One raw row from the JSONL dataset (link, channel, tags, descriptions) |
| `NormalizedEvent` | `ingest/models.py` | Cleaned version: stable `event_id`, resolved text, tag list, index flag |
| `EventAttributes` | `schemas/event_attributes.py` | Rule-extracted signals: free/paid, city, noise, type hints, dates |
| `ParsedHardConstraints` | `schemas/parsed_preferences.py` | User's strict requirements (budget, city, forbidden categories) |
| `ParsedSoftPreferences` | `schemas/parsed_preferences.py` | User's tastes (liked types, noise tolerance) |
| `ParsedUserPreferences` | `schemas/parsed_preferences.py` | Container: hard + soft + one-line summary |

---

## Module-by-module breakdown

---

### `ingest/` — raw data → NormalizedEvent

#### `ingest/models.py`
Defines the two core Pydantic models for ingest:

- **`RawEventRecord`** — maps directly to one line of `events_dataset.jsonl`. Fields: `link`, `source_channel`, `tags` (comma-separated string), `links` (dict for placeholder resolution), `event_description`, `ru_event_description`.
- **`NormalizedEvent`** — the cleaned output. Key additions vs raw: `event_id` (stable 16-char hex hash of the link), `tags` (list, not string), both `_raw` and `_resolved` versions of both descriptions, and `include_in_index` / `exclude_reason` from the heuristic filter.

#### `ingest/ids.py`
Single function `stable_event_id(link)`. Takes the event's URL, hashes it with SHA-256, returns the first 16 hex characters. This is the primary key for every event throughout the system — if the same URL appears twice, it gets the same ID.

#### `ingest/tags.py`
Single function `normalize_tags(tags_string)`. Splits a comma-separated tag string (e.g. `"#concert, #jazz, #openair"`), strips the `#`, lowercases, deduplicates, and returns a list. Output: `["concert", "jazz", "openair"]`.

#### `ingest/placeholders.py`
Single function `resolve_placeholders(text, links)`. The raw event descriptions sometimes contain tokens like `${PLACEHOLDER_0}` instead of actual URLs or titles. This function replaces those tokens with the `link_title` values from the `links` dict. If no links dict is provided, it just strips the placeholders.

#### `ingest/heuristics.py`
Single function `classify_for_retrieval_index(...)`. Decides whether an event row is worth including in semantic retrieval. Returns `(include: bool, reason: str | None)`.

It filters out:
- Text shorter than 80 characters (`"too_short"`)
- "Good morning" posts, recipe posts, weekly digest roundups (keyword matching)
- Events tagged `#promo`
- Posts that list 3+ weekdays — those are weekly schedule digests, not single events (`"digest_week_pattern"`)

The result is stored as `include_in_index` on `NormalizedEvent`.

#### `ingest/pipeline.py`
The main ingest logic. Three public functions:

- **`parse_jsonl_line(row_index, line)`** — processes one text line end-to-end: JSON parse → validate as `RawEventRecord` → normalize tags → resolve placeholders → run heuristic filter → return `NormalizedEvent`.
- **`iter_normalized_events(path)`** — generator that streams a whole JSONL file, yielding valid `NormalizedEvent` objects.
- **`ingest_jsonl_file(input_path, output_path)`** — reads the raw JSONL, writes normalized JSONL, returns `(written, skipped)` counts.

---

### `schemas/` — shared Pydantic models

#### `schemas/event_attributes.py`
Defines **`EventAttributes`** — the structured representation of what a rule-based pass extracted from one event's text and tags:

| Field | Type | Meaning |
|-------|------|---------|
| `event_id` | str | Links back to `NormalizedEvent` |
| `price_free_signal` | bool | Detected "free entry" language |
| `price_paid_signal` | bool | Detected ticket pricing language |
| `price_amount_rsd` | int\|None | Highest RSD amount found in text |
| `price_amount_eur` | int\|None | Highest EUR amount found in text |
| `city_hint` | enum | `belgrade`, `novi_sad`, `serbia_other`, `unknown` |
| `date_snippets` | list[str] | Up to 5 raw date strings found in text |
| `type_hints` | list[str] | Event category tokens (e.g. `concert`, `hiking`, `exhibition`) |
| `noise_level_hint` | enum | `low`, `medium`, `high`, `unknown` |
| `outdoor_hint` | bool\|None | True/False/None (unknown) |
| `extraction_method` | str | Audit trail (`"rules_v1"` or `"rules_v1_gemini_lazy"`) |

#### `schemas/parsed_preferences.py`
Defines the structured user preference profile that Gemini produces from free text (or that's loaded directly from `fixtures/synthetic_users.json`):

- **`ParsedHardConstraints`** — things that eliminate events: city filter, max budget in RSD, must-be-free flag, forbidden categories (e.g. `["techno", "rave"]`), time restrictions.
- **`ParsedSoftPreferences`** — things that adjust scores: `liked_types` (e.g. `["hiking", "jazz"]`), `noise_tolerance` (`low`/`medium`/`high`/`unknown`), `ok_paid_eur`, preferred time windows.
- **`ParsedUserPreferences`** — combines both, plus a `summary_one_line` field.

#### `schemas/gemini_event_extraction.py`
Pydantic schema used when Gemini is asked to extract event attributes via LLM (the lazy/optional path). Mirrors `EventAttributes` but structured for Gemini's JSON schema output.

---

### `extract/` — NormalizedEvent → EventAttributes

#### `extract/rules.py`
Pure regex and keyword functions — no ML, no external calls:

- `detect_free_entry(text)` — regex for "free entry", "besplatan", "вход свободн", etc.
- `max_rsd_amount(text)` / `max_eur_amount(text)` — find all price mentions in RSD or EUR, return the highest.
- `extract_date_snippets(text)` — find date patterns (DD.MM.YYYY, "Jan 15, 2025", etc.), return up to 5.
- `detect_city_hint(text)` — check for "Belgrade", "Beograd", "Novi Sad", etc.
- `paid_language_hint(text)` — detect "tickets from X RSD" phrasing.

#### `extract/attributes.py`
Builds a complete `EventAttributes` from a `NormalizedEvent` using the rules above. Main function: `extract_event_attributes(event)`.

Internal logic:
1. Concatenates English description (resolved + raw) into one text blob.
2. Calls all the rule functions from `rules.py`.
3. Maps tags to type hints using `_TAG_TO_TYPES` dict (e.g. `"techno" → "techno"`, `"livemusic" → "live_music"`).
4. Also scans description text for keyword matches (jazz, techno, hiking, etc.) in case they're not in tags.
5. Derives `noise_level_hint` from noisy tags (techno, rave, openair, party → `"high"`) vs quiet tags (lecture, museum → `"low"`).
6. Derives `outdoor_hint` from outdoor tags vs indoor tags.

Also provides **`extract_event_attributes_maybe_lazy_gemini(event)`** — same as above, but if env var `USE_LAZY_GEMINI_EVENT_ATTRIBUTES=1` is set AND the rule-based result has `noise_level_hint == "unknown"`, it makes one Gemini API call to fill in the gaps. The rule-based price/city signals are kept; Gemini only enriches noise, outdoor, types, dates.

---

### `retrieve/` — semantic search

#### `retrieve/text.py`
Single function `build_embedding_text(event)`. Constructs the string that gets embedded for each event:
```
channel: sta_imas_beograd
tags: concert, jazz, outdoor
description: <resolved English description, up to 2500 chars>
```
This is what the embedding model "sees" — channel and tags are included so the vector captures more signal than just the description alone.

#### `retrieve/encoder.py`
**`SentenceEncoder`** class. Lazy wrapper around `sentence-transformers`. The model (`all-MiniLM-L6-v2` by default, overridable via env var `SENTENCE_TRANSFORMER_MODEL`) is downloaded only on the first call to `encode()`. Returns L2-normalized float32 numpy arrays — normalized means cosine similarity = dot product, which makes the math in `cosine_search.py` simple.

#### `retrieve/cosine_search.py`
Single function `cosine_top_k(corpus_embeddings, query_embedding, k)`. Pure numpy — no external libraries. Computes dot products between the query vector and all corpus vectors (which works as cosine similarity since both are L2-normalized), then returns the top-k `(index, score)` pairs sorted best first.

#### `retrieve/faiss_index.py`
On-disk FAISS index management. Functions:

- `save_flat_ip_index(vectors, event_ids, output_dir, model_name)` — writes 3 files: `events.faiss` (the FAISS IndexFlatIP), `event_ids.json` (ordered list of event IDs matching vector rows), `manifest.json` (model name, dimensions, count).
- `load_manifest()`, `load_event_ids()`, `load_index()` — read those files back.
- `search(output_dir, query_vector, k)` — run a query against the on-disk index, return `(row_index, score)` pairs.

FAISS is faster than in-memory cosine search for large corpora, but both give the same results for `IndexFlatIP` on normalized vectors.

#### `retrieve/hard_filters.py`
Applies `ParsedHardConstraints` against `EventAttributes` to decide if an event should be kept after semantic retrieval:

- `passes_hard_constraints(hard, event, attrs)` — single user. Checks city, budget/free, forbidden categories.
- `passes_all_hard_constraints(hards, event, attrs)` — group version: event must pass every member's hard constraints (intersection rule — if anyone's constraint is violated, the event is dropped).

**City rule:** if user wants `belgrade`, only `novi_sad` events are filtered out (everything else is ambiguous enough to keep).
**Budget rule:** if `must_be_free` is set, event must have `price_free_signal=True` AND no paid signals. If `must_be_free_or_under_budget` with a cap, checks RSD directly and converts EUR at 1 EUR ≈ 120 RSD.
**Forbidden categories:** checks both tags and type_hints for substring matches.

#### `retrieve/service.py`
The high-level retrieval API — what the scripts, API, and UI all call. Key functions:

- `load_normalized_events(path)` — loads a JSONL or SQLite file into `list[NormalizedEvent]`.
- `retrieve_top_k(events, query_text, top_k)` — pure semantic search, no filtering.
- `retrieve_top_k_faiss(events_by_id, index_dir, query_text, top_k)` — same but using on-disk FAISS index.
- **`retrieve_semantic_then_hard_filter(...)`** — the main single-user pipeline: fetch a wide semantic pool (default 50), then apply hard constraints, return up to `result_k` events.
- **`retrieve_semantic_then_group_hard_filter(...)`** — same but applies all members' hard constraints (intersection). Used for groups.
- `retrieve_from_jsonl_file(...)` / `retrieve_from_jsonl_with_hard(...)` — convenience wrappers that load the file then call the above.

#### `retrieve/vector_index.py`
Notes/stub for future integration with hosted vector databases (Qdrant, pgvector). Not used in the current pipeline.

---

### `rank/` — scoring and group aggregation

#### `rank/soft_score.py`
Single function `soft_adjusted_score(prefs, event, attrs, semantic_similarity)`. Takes a user's soft preferences and adjusts the raw semantic similarity score:

- **Base score** = semantic similarity (clamped to [0, 1]).
- **Bonus** = +0.06 per matched `liked_type` (e.g. if user likes "jazz" and event has jazz tag), capped at +0.28 total.
- **Penalty** = -0.12 if user has low noise tolerance and event is high-noise; -0.04 if user is low-noise and event is medium-noise.

Result is a float in [0, 1].

#### `rank/group_rank.py`
Group ranking logic:

- **`per_user_soft_scores(members, event, semantic_similarity)`** — runs `soft_adjusted_score` for each group member, returns a list of scores.
- **`rank_by_least_misery(candidates, members)`** — the core algorithm. For each candidate event, computes every member's soft score, then sorts by `(min_score, mean_score)` descending. **Least-misery** means: prefer events where even the least-happy member gets a decent score. Returns tuples of `(event, semantic_sim, min_soft, mean_soft)`.
- `fixture_user_to_prefs(user_dict)` — converts a `synthetic_users.json` entry to `ParsedUserPreferences`.
- `combined_group_query_text(users)` — concatenates all members' `preferences_plain_text` into one query string for the semantic search step.

---

### `llm/` — Gemini integration (optional)

All LLM features require `pip install -e ".[gemini]"` and `GEMINI_API_KEY`.

#### `llm/prompts.py`
Three system prompt strings used across the LLM calls:

- `PREFERENCE_SYSTEM_INSTRUCTION` — tells Gemini how to map free-text user preferences to `ParsedUserPreferences` JSON fields. Specifies rules: hard vs soft split, snake_case tokens, leave unknown fields empty.
- `RANKING_EXPLAIN_INSTRUCTION` — tells Gemini to write a 2–5 paragraph plain-language explanation of why events were recommended. Rules: match user's language (Serbian if preference text is Serbian), don't invent URLs, don't claim certainty about prices.
- `EVENT_ATTRIBUTES_LLM_INSTRUCTION` — tells Gemini to extract `EventAttributes` JSON from an event description. Used only in the lazy enrichment path.

#### `llm/gemini_preferences.py`
Function `parse_preferences_plain_text(text)`. Takes a user's free-text description of what they want (e.g. "Hoću nešto besplatno u centru, posle 18h, bez techno žurki") and returns a structured `ParsedUserPreferences` from Gemini.

How it works:
1. Builds `GenerateContentConfig` with `response_mime_type="application/json"` and the Pydantic JSON schema.
2. Sends the text to `gemini-2.0-flash` (configurable via `GEMINI_MODEL` env var).
3. Validates the JSON response against `ParsedUserPreferences`.

The `_generation_config` helper handles API version differences — the parameter name for schema constraints changed between `google-genai` releases.

#### `llm/gemini_explain.py`
Function `explain_ranking_plain(preferences_context, ranked_lines)`. After ranking is done, generates a human-readable explanation. Takes:
- `preferences_context` — JSON dump of the user's or group's preferences.
- `ranked_lines` — list of formatted strings describing the top events.

Helper functions `ranked_lines_from_single` and `ranked_lines_from_group` format the ranked results into those strings.

#### `llm/gemini_event_attributes.py`
`parse_event_attributes_with_gemini(event)`. Used only when `USE_LAZY_GEMINI_EVENT_ATTRIBUTES=1` env var is set and rules couldn't determine noise level. Makes a single Gemini call to get `EventAttributes`-like JSON for one event. The result is merged back in `extract/attributes.py` — rule-based price/city always wins; Gemini fills in noise, outdoor, types.

---

### `storage/` — SQLite persistence

#### `storage/sqlite_store.py`
Two tables: `normalized_events` (full payload as JSON + indexed columns) and `event_attributes` (rule-extracted signals as typed columns).

Key functions:
- `bulk_load_normalized_jsonl(jsonl_path, db_path)` — reads normalized JSONL, extracts attributes, inserts everything into SQLite. Called by `scripts/load_jsonl_to_sqlite.py`.
- `load_all_normalized_events(db_path)` — reads back all events ordered by `row_index`. Called by `retrieve/service.py` when the corpus path ends in `.db` or `.sqlite`.
- `load_indexable_event_ids(db_path)` — returns only event IDs with `include_in_index=1`, used by `scripts/build_vector_index.py`.

The design stores the full `NormalizedEvent` as a JSON payload column so it can be reconstructed without joins, while also having typed columns for SQL filtering (useful if future work adds SQL-level pre-filtering).

---

### `eval/` — offline quality measurement

#### `eval/metrics.py`
Three standard IR metrics, given a set of human-labeled relevant event IDs and a ranked list of retrieved IDs:

- **`precision_at_k(relevant, ranked_ids, k)`** — of the top-k results, what fraction are relevant? (0 to 1)
- **`recall_at_k(relevant, ranked_ids, k)`** — of all relevant events, what fraction appear in the top-k? (0 to 1)
- **`reciprocal_rank(relevant, ranked_ids)`** — 1 divided by the rank position of the first relevant hit. Measures how high the first correct result appears.

These run against `fixtures/eval_labels.json` via `scripts/run_eval.py`.

---

### `api/` — FastAPI REST service

#### `api/paths.py`
Security helper. `resolve_under_repo(root, raw_path)` ensures any file path sent in an API request resolves to something inside the repo root — prevents path traversal attacks (e.g. a request asking to read `../../etc/passwd`). The repo root is set via `BELGRADE_RECOMMENDER_ROOT` env var (defaults to current working directory).

#### `api/main.py`
The FastAPI application. Three endpoints:

**`GET /health`** — returns `{"status": "ok"}`. Used by Docker healthchecks.

**`POST /v1/recommendations/single`** — single-user recommendations. Request body:
```json
{
  "events_path": "fixtures/events_smoke.jsonl",
  "fixture_user_id": "ana",
  "use_hard": true,
  "top_k": 5,
  "include_explanation": false
}
```
Loads the fixture user's preferences, runs semantic + hard-filter retrieval, returns scored event list. If `include_explanation: true`, calls Gemini after ranking.

**`POST /v1/recommendations/group`** — group recommendations. Takes a `group_id`, loads all member users, combines their queries, runs group hard-filter retrieval + least-misery ranking.

CORS is configured to allow the Streamlit UI (port 8501) to call this API.

---

### `ui/` — Streamlit browser interface

#### `ui/streamlit_app.py`
Browser-based demo with two tabs:

- **Single user tab** — pick a fixture user, optionally a FAISS index dir, set top-k, run retrieval. Shows scored results with links and description previews.
- **Group tab** — pick a group from the fixture file, runs group hard-filter + least-misery ranking. Shows semantic score, min-soft, and mean-soft for each result.

Path inputs are validated with the same repo-root security check as the API. Results are cached with `@st.cache_data` to avoid re-embedding on each UI interaction.

---

### `scripts/` — CLI entrypoints

| Script | What it does |
|--------|-------------|
| `ingest_events.py` | Runs `ingest_jsonl_file` — raw JSONL → normalized JSONL |
| `extract_event_attributes.py` | Runs `extract_event_attributes` on every line → structured JSONL |
| `load_jsonl_to_sqlite.py` | Calls `bulk_load_normalized_jsonl` → SQLite database |
| `build_smoke_dataset.py` | Samples 150 indexable events from normalized JSONL → `fixtures/events_smoke.jsonl` |
| `build_vector_index.py` | Embeds all indexable events, writes FAISS index to `data/processed/vectors/` |
| `demo_retrieval.py` | CLI demo of semantic (and optionally hard-filtered) retrieval for one fixture user |
| `demo_group_rank.py` | CLI demo of the full group pipeline |
| `parse_preferences_demo.py` | Calls Gemini to parse free-text preferences → prints structured JSON |
| `extract_event_attributes_llm_demo.py` | Compares rule-based vs Gemini attribute extraction on sample events |
| `run_eval.py` | Loads `fixtures/eval_labels.json`, runs retrieval, prints P@k / R@k / MRR |

---

### `fixtures/` — test data

| File | What it contains |
|------|-----------------|
| `events_smoke.jsonl` | 150 pre-sampled `NormalizedEvent` rows — used by all demos without needing the full dataset |
| `synthetic_users.json` | 4 fictional users (ana, marko, jelena, stefan) with `preferences_plain_text`, `hard_constraints`, `soft_preferences`, plus 2 groups (`weekend_group`, `family_outing`) |
| `eval_labels.json` | Human-labeled relevance cases for offline evaluation; needs to be filled in manually |

---

## Dependency map (which module calls which)

```
scripts / api / ui
    │
    ├── retrieve/service.py ──► retrieve/encoder.py
    │       │                       └─► sentence-transformers
    │       ├──► retrieve/cosine_search.py
    │       ├──► retrieve/faiss_index.py ──► faiss-cpu
    │       ├──► retrieve/hard_filters.py
    │       └──► extract/attributes.py ──► extract/rules.py
    │                   └─ (optional) ──► llm/gemini_event_attributes.py
    │
    ├── rank/group_rank.py ──► rank/soft_score.py
    │
    ├── llm/gemini_preferences.py ──► llm/prompts.py
    ├── llm/gemini_explain.py     ──► llm/prompts.py
    │
    └── storage/sqlite_store.py
```

All data models (schemas) are imported by almost every module — they are the shared contract between layers.
