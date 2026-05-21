# Ingest & Schema Improvements

---

## Part 1 — `ingest/heuristics.py`

### 1. Bug: Wildcard `*` in `EXCLUDE_SUBSTRINGS`

**Entry:** `"events* in the week"`

Python's `in` operator does literal string matching — the `*` is not a wildcard.
This phrase will never match anything.

**Fix options:**
- Remove the `*` and use `"events in the week"` (if that exact phrase is sufficient)
- Or switch that specific check to a regex:

```python
import re
DIGEST_PATTERNS = [
    re.compile(r"events?\s+in\s+the\s+week", re.IGNORECASE),
]
# then in classify_for_retrieval_index:
for pattern in DIGEST_PATTERNS:
    if pattern.search(combined) or pattern.search(raw_lower):
        return False, "digest_pattern"
```

---

### 2. `digest_week_pattern` check is brittle

**Current logic:** counts weekday names that are immediately followed by a comma (`"monday,"`, `"tuesday,"`, etc.).

This will miss digests formatted as:
- `"Monday evening"` — no comma
- `"Monday - Friday"` — dash separator
- `"понедельник, вторник"` — Serbian/Russian weekday names

**Fix:** broaden the match to include weekday names without requiring a trailing comma, and add BCS/RU weekday names:

```python
WEEKDAYS_EN = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
WEEKDAYS_SR = ("ponedeljak", "utorak", "sreda", "četvrtak", "petak", "subota", "nedelja")
WEEKDAYS_RU = ("понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье")

ALL_WEEKDAYS = WEEKDAYS_EN + WEEKDAYS_SR + WEEKDAYS_RU

weekday_hits = sum(1 for w in ALL_WEEKDAYS if w in lower)
if weekday_hits >= 3:
    return False, "digest_week_pattern"
```

---

### 3. `EXCLUDE_SUBSTRINGS` is sparse — no Serbian/BCS noise coverage

The list handles EN and RU noise phrases but nothing in Serbian.
Common noise posts in this dataset likely include Serbian equivalents.

**Suggested additions:**

```python
# Serbian noise phrases
"dobro jutro",          # already there
"recepti:",             # recipe posts
"sastojci:",            # "ingredients:" in Serbian
"zagrejte rernu",       # "preheat the oven"
"broj dana",            # "number of the day"
"vest dana",            # "news of the day"
"nedeljni pregled",     # "weekly overview"
"događaji za nedelju",  # "events for the week"
```

---

### 4. No positive signal requirement

The filter is purely exclusionary — anything that isn't caught by the rules passes through.
Adding a lightweight positive-signal check would tighten the index.

**Idea:** require at least one of the following to be present:
- A time pattern: `re.search(r'\b\d{1,2}[:.]\d{2}\b', text)` — e.g. `20:00`, `8.30`
- A date pattern: `re.search(r'\b\d{1,2}\.\s?\d{1,2}\.', text)` — e.g. `15.5.` or `15. 5.`
- A venue keyword: `("venue", "club", "bar", "hall", "kafana", "klub", "sala", "bašta")`

This could be a separate, stricter mode — e.g. `require_positive_signal: bool = False` — so it doesn't break existing behavior.

---

### 5. Minor: `combined` includes resolved description + tags but not raw

The loop checks both `combined` and `raw_lower` separately, which is correct.
Just worth noting: if `event_description_resolved` and `event_description_raw` are similar in most cases, the double-check may rarely matter. If they can diverge significantly, consider documenting why both are checked.

---

## Part 2 — `schemas/event_attributes.py`

### 6. `NoiseHint` includes `"medium"` but nothing ever produces it

`_noise_hint()` in `attributes.py` returns only `"high"`, `"low"`, or `"unknown"`.
The `"medium"` value is defined in the `Literal` but is dead code — either add logic that produces it or remove it.

---

### 7. `type_hints` has no `max_length` in the schema

The merge logic in `attributes.py` caps `type_hints` at 12, but the schema field itself has no constraint.
Anything bypassing the merge can produce an unbounded list.

**Fix:**
```python
type_hints: list[str] = Field(
    default_factory=list,
    max_length=12,
    description="Merged snake_case hints from tags + keywords (deduped).",
)
```

---

### 8. `extraction_method` is a free-form `str`

Three concrete values are used across the codebase (`"rules_v1"`, `"rules_v1_gemini_lazy"`).
Making it a `Literal` catches typos at parse time:

```python
ExtractionMethod = Literal["rules_v1", "rules_v1_gemini_lazy"]

extraction_method: ExtractionMethod = Field(default="rules_v1", ...)
```

---

### 9. `price_amount_rsd / eur` uses `int` — decimal prices silently truncated

`_parse_amount` in `rules.py` strips decimals. For RSD this is usually fine, but EUR prices like `"15.50 EUR"` get parsed by concatenating digit parts — potentially producing `1550` instead of `15`.
Using `float` would be safer, or at minimum the truncation behavior should be documented.

---

### 10. `price_free_signal` and `price_paid_signal` can both be `True`

This is sometimes valid ("free for members, 500 RSD for guests") but is an easy source of confusion downstream.
A `model_validator` would at least document the intent:

```python
from pydantic import model_validator

@model_validator(mode="after")
def _check_price_conflict(self) -> "EventAttributes":
    # Both True is valid (tiered pricing), but flag it for awareness
    return self
```

Or add a derived property `is_tiered_pricing: bool` to make the dual-True case explicit.

---

### 11. `CityHint` is duplicated across files

`rules.py:8` redefines `CityHint = Literal[...]` independently instead of importing from `event_attributes.py`.
If the two ever diverge, you get a silent type mismatch.

**Fix:** remove the definition from `rules.py` and import from the schema:
```python
from belgrade_recommender.schemas.event_attributes import CityHint
```

---

### 12. Missing fields worth considering

| Field | Why useful |
|-------|-----------|
| `min_price_rsd / min_price_eur` | `max_rsd_amount` only stores the ceiling. Budget filtering needs the floor (e.g. "500–1500 RSD"). |
| `recurrence_hint` | Weekly recurring events vs one-time events are meaningfully different for recommendations. |
| `audience_hint` | Family-friendly vs adults-only is a common recommendation dimension. |

---

## Part 3 — `extract/rules.py` and `extract/attributes.py`

### 13. `_parse_amount` corrupts decimal EUR prices

For input `"15.50"`, the function checks if the last part after `.` has 3 digits — it doesn't (2 digits), so it falls into the second branch and concatenates `"15"` + `"50"` → `1550`. A `15.50 EUR` ticket gets stored as `1550 EUR`.

**Fix:** use `float` and round, or handle 2-digit decimals explicitly:
```python
def _parse_amount(raw: str) -> float | None:
    s = raw.replace(" ", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None
```
Also update `price_amount_rsd / eur` in the schema to `float | None`.

---

### 14. `_DATE` regex misses yearless dates

The pattern only matches dates with a full 4-digit year (`20\d{2}`). Dates like `"15.5."` or `"subota, 17. maja"` — very common in local event posts — are completely missed.

**Fix:** add a yearless pattern:
```python
_DATE = re.compile(
    r"\b\d{1,2}[./-]\d{1,2}[./-]20\d{2}\b|"
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+20\d{2}\b|"
    r"\b\d{1,2}\.\s?\d{1,2}\.\b",  # yearless: "15.5." or "15. 5."
    re.IGNORECASE,
)
```

---

### 15. `_noise_hint` ignores event text — only checks tags

An event with no tags but description containing `"techno night"` or `"rave party"` gets `noise_level_hint = "unknown"`. The keyword loop in `extract_event_attributes` enriches `type_hints` from text, but noise level stays tag-only.

**Fix:** extend `_noise_hint` to also scan text:
```python
_NOISY_KEYWORDS = frozenset({"techno", "rave", "nightlife", "party", "dj set"})
_QUIET_KEYWORDS = frozenset({"lecture", "exhibition", "classical", "reading", "workshop"})

def _noise_hint(tags: list[str], text_lower: str = "") -> NoiseHint:
    lowered = {t.lower() for t in tags}
    if lowered & _NOISY_TAGS or any(k in text_lower for k in _NOISY_KEYWORDS):
        return "high"
    if lowered & _QUIET_TAGS or any(k in text_lower for k in _QUIET_KEYWORDS):
        return "low"
    return "unknown"
```
Also update the call site in `extract_event_attributes` to pass `text_lower`.

---

### 16. `_outdoor_hint` misses `"open air"` (no hyphen)

Currently checks for `"open-air"` but not `"open air"`. Both forms appear in real descriptions.

**Fix:**
```python
if lowered & _OUTDOOR_TAGS or "outdoor" in text_lower or "open-air" in text_lower or "open air" in text_lower:
    return True
```

---

### 17. `detect_city_hint` calls `.lower()` twice on the same string

Line 84: `"белград" in (text or "").lower()` — but `t` is already `text.lower()`. Minor inefficiency but easy to clean up.

**Fix:** replace with `"белград" in t`.

---

### 18. `paid_language_hint` is largely redundant with the amount extractors

In `attributes.py`, `paid_hint = text_rules.paid_language_hint(text) and not free`, then `price_paid_signal = paid_hint or bool(rsd or eur)`. The `paid_language_hint` regex also matches `\d+\s*(?:RSD|EUR|€)` — which is almost identical to what `_RSD` and `_EUR` already catch. The only non-redundant value it adds is `"tickets from"` / `"ticket price"` wording.

Consider either removing `paid_language_hint` and just relying on the amount extractors, or narrowing it to only the ticket-wording patterns it uniquely covers.

---

### 19. `type_hints` keyword fallback in `extract_event_attributes` is too narrow

The keyword loop (lines 98–106 in `attributes.py`) only covers 5 keywords: `jazz`, `techno`, `hiking`, `exhibition`, `lecture`. But `_TAG_TO_TYPES` has 19 mappings. Events described in text with `"concert"`, `"festival"`, `"workshop"`, `"film"` etc. won't get type hints unless they also have matching tags.

**Fix:** drive the text keyword loop from `_TAG_TO_TYPES` itself so they stay in sync:
```python
for keyword, label in _TAG_TO_TYPES.items():
    if keyword in text_lower and label not in type_hints:
        type_hints.append(label)
```

---

## Part 4 — `retrieve/` folder

### 20. `build_embedding_text` truncates mid-word (`text.py`)

`text[:max_chars]` can slice mid-word or mid-sentence, corrupting the last token the encoder sees.

**Fix:** truncate at the last space before the limit:
```python
if len(text) > max_chars:
    text = text[:max_chars].rsplit(" ", 1)[0]
```

---

### 21. Structured signals not included in embedding text (`text.py`)

`type_hints`, `city_hint`, `noise_level_hint` from `EventAttributes` are computed separately but never woven into the embedding string. A query like `"jazz event this weekend"` has no signal to match against unless the description happens to mention jazz.

**Idea:** optionally prepend a signals line when attrs are available:
```python
# e.g. "type: concert, jazz | city: belgrade | noise: high"
```
This is a meaningful retrieval quality improvement, not just a cleanup.

---

### 22. Incomplete docstring in `build_embedding_text` (`text.py`)

The docstring ends mid-sentence: `"Uses resolved English description;"` — the sentence has no conclusion.

---

### 23. `_city_ok` logic is wrong for `"serbia_other"` preference (`hard_filters.py`)

If a user prefers `"serbia_other"` (events outside main cities), events in Belgrade still pass because the function returns `True` for any preference that isn't `"belgrade"`. This seems backwards — someone who wants a day trip outside the city shouldn't get Belgrade events.

**Fix:** make the logic explicit for each preference value instead of relying on fall-through `return True`.

---

### 24. `must_be_free=True` rejects events with no price info (`hard_filters.py`)

Lines 53–58: if `price_free_signal` is False but there are also no paid signals or amounts, the function returns `False`. An event with completely unknown pricing gets excluded from free-only results. This is too aggressive — unknown price should likely pass (or be a separate `"unknown"` bucket).

---

### 25. Hardcoded EUR→RSD rate of `120` (`hard_filters.py`)

No constant name, no comment. Will silently become wrong as the rate changes.

**Fix:**
```python
_EUR_TO_RSD_APPROX = 120  # update periodically
```

---

### 26. Bidirectional substring match causes false positives in `_forbidden_categories_ok` (`hard_filters.py`)

`token in tag or tag in token` means a forbidden category `"art"` blocks events tagged `"party"` (since `"art"` is a substring of `"party"`). Word-boundary or exact-equality matching would be safer:
```python
if token == tag or token in tag.split("_"):
    return False
```

---

### 27. `SentenceEncoder()` instantiated fresh on every call (`service.py`)

When `encoder=None`, `retrieve_top_k` creates a new encoder — model weights reload every call. For any real workload this is a serious performance problem.

**Fix:** module-level singleton:
```python
_default_encoder: SentenceEncoder | None = None

def _get_encoder() -> SentenceEncoder:
    global _default_encoder
    if _default_encoder is None:
        _default_encoder = SentenceEncoder()
    return _default_encoder
```

---

### 28. `extract_event_attributes_maybe_lazy_gemini` called inside filter loop (`service.py`)

In both `retrieve_semantic_then_hard_filter` and `retrieve_semantic_then_group_hard_filter`, attributes are recomputed from scratch for every event on every query. Pre-computing attrs once per event (keyed by `event_id`) and passing a cache in would be a significant speedup for repeated queries against the same corpus.

---

### 29. Silent parse error swallowing in `load_normalized_events_jsonl` (`service.py`)

The bare `except Exception: continue` hides all data issues with no trace. Should at minimum log a warning with the line content or error.

---

### 30. Two near-identical retrieve functions (`service.py`)

`retrieve_semantic_then_hard_filter` and `retrieve_semantic_then_group_hard_filter` are identical except for one function call. Could be unified:
```python
def retrieve_semantic_then_filter(
    ...,
    hards: Sequence[ParsedHardConstraints],
) -> list[tuple[NormalizedEvent, float]]:
    ...
    if passes_all_hard_constraints(hards, event, attrs):
        ...
```
The single-constraint variant becomes a one-liner wrapper.

---

### 31. `SentenceEncoder` loads weights twice for the same model name (`encoder.py`)

`_model` is instance-level, so two `SentenceEncoder("all-MiniLM-L6-v2")` instances each load independently.

**Fix:** class-level cache:
```python
_cache: dict[str, Any] = {}

def encode(self, texts: list[str]) -> np.ndarray:
    if self.model_name not in SentenceEncoder._cache:
        SentenceEncoder._cache[self.model_name] = SentenceTransformer(self.model_name)
    model = SentenceEncoder._cache[self.model_name]
    ...
```

---

### 32. Default model is English-only for a multilingual corpus (`encoder.py`)

`"all-MiniLM-L6-v2"` is trained on English. The corpus mixes EN, Serbian, and Russian. A multilingual model like `"paraphrase-multilingual-MiniLM-L12-v2"` would give meaningfully better semantic matching across languages — worth benchmarking.

---

### 33. `faiss_index.search` reloads the index from disk on every call (`faiss_index.py`)

`search()` calls `load_index()` which reads the `.faiss` file from disk every time. For repeated queries this is unnecessarily slow.

**Fix:** expose a `load_index` call separately so callers can cache the loaded index object and pass it directly into search, or add an optional `index` parameter:
```python
def search(
    output_dir: Path,
    query_vector: np.ndarray,
    k: int,
    *,
    index: Any | None = None,
) -> list[tuple[int, float]]:
    index = index or load_index(output_dir)
    ...
```

---

### 34. `vector_index.py` is a placeholder that exports nothing

The file is only a comment block. It adds noise to the module structure without contributing code. Either remove it or replace it with actual shared types/constants used across the retrieve layer.

---

## Part 5 — `rank/soft_score.py` and `rank/group_rank.py`

### 35. Bidirectional substring match for liked types (`soft_score.py`)

`any(lt in x or x in lt for x in tag_hint)` — same false-positive pattern as in `hard_filters.py`. A liked type of `"art"` boosts events tagged `"party"`.

---

### 36. Magic numbers for bonus cap and step (`soft_score.py`)

`bonus += 0.06` and `bonus = min(bonus, 0.28)` are unexplained constants. No names, no comment on the rationale (0.28 = 4.67 × 0.06). Hard to tune deliberately.

**Fix:**
```python
_LIKED_TYPE_BONUS = 0.06
_MAX_LIKED_BONUS = 0.28
```

---

### 37. Dead noise penalty branch for `"medium"` (`soft_score.py`)

`if noise == "low" and attrs.noise_level_hint == "medium": penalty += 0.04` — as established in issue #6, `_noise_hint()` never produces `"medium"`. This penalty never fires.

---

### 38. Clamping semantic similarity to `[0, 1]` discards negative signal (`soft_score.py`)

`base = max(0.0, min(1.0, float(semantic_similarity)))` — cosine similarity for normalized vectors is in `[-1, 1]`. A negative similarity means the event is actively dissimilar. Clamping to 0 makes a poor match indistinguishable from a neutral one before bonuses are applied — a bad event can be rescued by soft bonuses when it shouldn't be.

---

### 39. No outdoor preference signal in soft scoring (`soft_score.py`)

`attrs.outdoor_hint` exists and is computed, but `soft_adjusted_score` never reads it. If the user has an outdoor preference, it has no effect on ranking.

---

### 40. `extract_event_attributes_maybe_lazy_gemini` called per event during ranking (`group_rank.py`)

Same issue as #28 — attrs are recomputed per event per ranking call. Should be passed in or cached by `event_id`.

---

### 41. Locked into one aggregation strategy (`group_rank.py`)

`rank_by_least_misery` hardcodes the min-score strategy with no way to switch to average, Borda count, or fairness-weighted approaches. The function name bakes in the strategy permanently, making experimentation hard.

---

### 42. `fixture_user_to_prefs` and `combined_group_query_text` don't belong in rank (`group_rank.py`)

`fixture_user_to_prefs` is a test utility ("fixture" in the name signals this). `combined_group_query_text` is query construction, not ranking. Both are imported directly by `api/main.py`, confirming they belong in a shared utility or API layer.

---

### 43. 4-tuple return type of `rank_by_least_misery` is opaque (`group_rank.py`)

`(event, semantic_sim, min_soft, mean_soft)` by position is easy to misread at callsites. A named structure would be self-documenting:
```python
from dataclasses import dataclass

@dataclass
class RankedEvent:
    event: NormalizedEvent
    semantic_sim: float
    min_soft: float
    mean_soft: float
```

---

## Part 6 — `llm/` folder

### 44. `_gemini_available()` duplicated in three files

Identical function defined in `gemini_preferences.py`, `gemini_explain.py`, and `gemini_event_attributes.py`. Should be a single shared utility, e.g. in `llm/__init__.py` or a `llm/_client.py`.

---

### 45. `_generation_config()` duplicated in two files

Nearly identical function in `gemini_preferences.py` and `gemini_event_attributes.py`. Same fix — extract to a shared module.

---

### 46. `genai.Client` created fresh on every LLM call

A new `genai.Client(api_key=key)` is instantiated in every call to `parse_preferences_plain_text`, `parse_event_attributes_with_gemini`, and `explain_ranking_plain`. The client is stateless and safe to reuse — cache it at module level keyed by API key.

---

### 47. No retry logic for transient Gemini API failures

Rate limit errors, network timeouts, and transient 5xx responses are not retried — they surface as unhandled exceptions. Even a simple exponential backoff with 2–3 attempts would make the system more robust.

---

### 48. `"gemini_v1"` extraction method not in the `extraction_method` Literal (if #8 is fixed)

`gemini_event_attributes.py:98` sets `extraction_method="gemini_v1"`, but if `extraction_method` is made a `Literal` (issue #8), this value must be added: `Literal["rules_v1", "rules_v1_gemini_lazy", "gemini_v1"]`.

---

### 49. `noise_tolerance` in prompt includes `"medium_high"` but `NoiseHint` doesn't (`prompts.py`)

`PREFERENCE_SYSTEM_INSTRUCTION` tells Gemini to use `"low | medium | medium_high | high | unknown"` for `noise_tolerance`. But `NoiseHint` in `event_attributes.py` only has `"low | medium | high | unknown"` — no `"medium_high"`. The soft scoring comparison `if noise in ("low", "unknown")` will silently not match `"medium_high"`, producing no penalty when it should.

Either add `"medium_high"` to `NoiseHint`, or remove it from the prompt.

---

### 50. `EVENT_ATTRIBUTES_LLM_INSTRUCTION` says "integers only" for prices (`prompts.py`)

The prompt instructs Gemini to produce `price_amount_rsd / price_amount_eur` as integers. This reinforces the decimal truncation bug (#9/13). If prices are fixed to `float`, the prompt must be updated too.

---

## Part 7 — `schemas/` folder

### 51. `CityHint` and `NoiseHint` defined a third time in `gemini_event_extraction.py`

`gemini_event_extraction.py:9-10` defines both types again independently. They're now in `event_attributes.py`, `rules.py`, and `gemini_event_extraction.py`. All three should import from one canonical location.

---

### 52. `GeminiEventAttributeExtraction` duplicates `EventAttributes` fields

Every field in `GeminiEventAttributeExtraction` is a copy of an `EventAttributes` field (minus `event_id` and `extraction_method`). Adding a new field to `EventAttributes` requires a manual matching addition here. Could use inheritance:
```python
class GeminiEventAttributeExtraction(EventAttributes):
    event_id: str = ""
    extraction_method: str = "gemini_v1"
```
Or at minimum add a test that checks field parity.

---

### 53. `NoiseTolerance` has `"medium_high"` but `NoiseHint` doesn't

`parsed_preferences.py:14` defines `NoiseTolerance = Literal["low", "medium", "medium_high", "high", "unknown"]`. `NoiseHint` in `event_attributes.py` has no `"medium_high"`. The soft scoring comparison in `soft_score.py` works on `NoiseTolerance` values vs `NoiseHint` values — a user with `noise_tolerance = "medium_high"` gets no penalty at all, because no branch matches it. This is the same underlying issue as #49.

---

## Part 8 — `ingest/` folder

### 54. `parse_jsonl_line` silently returns `None` for all errors (`pipeline.py`)

Invalid JSON, failed validation, and missing `link` all return `None` with no logging. Same silent-swallowing pattern as #29. Should at minimum log warnings in debug mode.

---

### 55. `RawEventRecord.tags` is a `str`, `NormalizedEvent.tags` is `list[str]` (`models.py`)

The string→list conversion happens implicitly via `normalize_tags` in the pipeline. This asymmetry is undocumented in the model and could confuse anyone building alternative ingest paths.

---

### 56. No `ingested_at` timestamp on `NormalizedEvent` (`models.py`)

No way to know how stale a record is, or to do time-based cache invalidation. A `ingested_at: datetime` field would be cheap to add and useful for debugging.

---

### 57. `normalize_tags` only handles comma-separated tags (`tags.py`)

If a future data source uses space or semicolon separation, the function silently produces wrong results (one token containing the whole string). The separator assumption should be documented or made configurable.

---

## Part 9 — `storage/sqlite_store.py`

### 58. `price_amount_rsd/eur` stored as `INTEGER` in SQLite schema

Same decimal bug as #9/13 — if the Python type is changed to `float`, the DB schema must change from `INTEGER` to `REAL`, which requires a migration.

---

### 59. Row-by-row insert in `bulk_load_normalized_jsonl`

Events and attributes are inserted one row at a time inside a `for` loop. For large corpora this is slow. `executemany` with batched data would be significantly faster.

---

### 60. `fetchall()` in `load_all_normalized_events` loads entire corpus into memory

For large datasets, iterating the cursor row-by-row would be more memory-efficient than `fetchall()`.

---

### 61. `date_snippets` not stored as a queryable column

`date_snippets` are in `EventAttributes` but the `event_attributes` table has no `date_snippets` column — they're not accessible from SQL. Any future date-based filtering from the DB layer would require re-parsing `payload_json`.

---

## Part 10 — `api/main.py`

### 62. Corpus reloaded from disk on every request

Both endpoints call `load_normalized_events` / `retrieve_from_jsonl_*` on every request with no in-memory caching. For any realistic load, the corpus should be loaded once at startup using FastAPI's lifespan or a module-level singleton.

---

### 63. `SentenceEncoder` recreated per request

Service functions create a new encoder when none is passed in — model weights reload on every API call (same as #27). The encoder should be initialized at startup and shared across requests.

---

### 64. Explanation errors silently swallowed with no logging

Both endpoints catch `(ImportError, OSError, RuntimeError, ValueError)` when calling Gemini and set `explanation = None` with no log entry. The caller gets no indication that explanation generation failed or why.

---

### 65. Path inputs accept arbitrary strings without detail on rejection

`body.events_path` and `body.fixture_path` are accepted as free strings. `resolve_under_repo` blocks traversal, but the error response is just `"Invalid path"` with no detail. Better to surface what failed (file not found vs. path escape).

---

## Part 11 — `eval/metrics.py`

### 66. Missing `ndcg_at_k` and `mean_reciprocal_rank`

`precision_at_k`, `recall_at_k`, and `reciprocal_rank` are implemented, but:
- `reciprocal_rank` is per-query only — a `mean_reciprocal_rank` over a list of queries is the standard use and is missing.
- `ndcg_at_k` (normalized discounted cumulative gain) is the standard metric for graded ranking quality and isn't present. For a recommender system, NDCG is more informative than P@k alone.

---

## Priority

| # | File | Issue | Severity | Effort |
|---|------|-------|----------|--------|
| 1 | heuristics.py | Wildcard `*` bug | High — phrase never matches | Trivial |
| 11/51 | rules.py + gemini_event_extraction.py | `CityHint`/`NoiseHint` defined 3× | High — silent divergence risk | Trivial |
| 9/13/58 | rules.py + schema + sqlite | `int` price corruption for EUR | High — data corruption across layers | Low |
| 14 | rules.py | `_DATE` misses yearless dates | High — most common local format | Low |
| 24 | hard_filters.py | `must_be_free` rejects unknown-price events | High — silently removes valid candidates | Low |
| 27/63 | service.py + api | Encoder recreated per call/request | High — major perf issue | Low |
| 28/40 | service.py + group_rank.py | Attrs recomputed inside loops | High — perf on repeated queries | Medium |
| 33 | faiss_index.py | Index reloaded from disk per query | High — perf issue | Low |
| 49/53 | prompts.py + parsed_preferences.py | `"medium_high"` noise mismatch | High — soft penalty silently skipped | Low |
| 62 | api/main.py | Corpus reloaded per request | High — perf issue | Medium |
| 15 | attributes.py | `_noise_hint` ignores text | Medium | Low |
| 19 | attributes.py | `type_hints` keyword loop too narrow | Medium | Trivial |
| 21 | text.py | Structured signals not in embedding | Medium — retrieval quality | Medium |
| 23 | hard_filters.py | `_city_ok` wrong for `"serbia_other"` | Medium — logic bug | Low |
| 26/35 | hard_filters.py + soft_score.py | Substring false positives | Medium | Low |
| 29/54 | service.py + pipeline.py | Silent error swallowing | Medium | Trivial |
| 31 | encoder.py | Model loaded twice for same name | Medium — perf | Low |
| 32 | encoder.py | English-only default model | Medium — retrieval quality | Low |
| 38 | soft_score.py | Clamping discards negative similarity | Medium — ranking quality | Low |
| 39 | soft_score.py | Outdoor preference unused in scoring | Medium | Low |
| 41 | group_rank.py | Locked into least-misery strategy | Medium — experimentation blocked | Medium |
| 44/45/46 | llm/*.py | Gemini helpers duplicated, client recreated | Medium — perf + maintainability | Low |
| 52 | gemini_event_extraction.py | `GeminiEventAttributeExtraction` duplicates `EventAttributes` | Medium — sync risk | Low |
| 2 | heuristics.py | Weekday comma dependency | Medium | Low |
| 3 | heuristics.py | Missing Serbian noise phrases | Medium | Low |
| 7 | event_attributes.py | `type_hints` no `max_length` | Medium | Trivial |
| 8/48 | event_attributes.py + gemini | `extraction_method` free-form str | Medium | Trivial |
| 47 | llm/*.py | No retry on Gemini failures | Medium | Low |
| 59 | sqlite_store.py | Row-by-row insert, no batching | Medium — perf at scale | Low |
| 20 | text.py | Mid-word truncation | Low | Trivial |
| 25 | hard_filters.py | Hardcoded EUR→RSD rate | Low | Trivial |
| 30 | service.py | Two near-identical retrieve functions | Low | Low |
| 34 | vector_index.py | Empty placeholder file | Low | Trivial |
| 36 | soft_score.py | Magic numbers in bonus logic | Low | Trivial |
| 37 | soft_score.py | Dead `"medium"` noise penalty | Low | Trivial |
| 42 | group_rank.py | Fixture utils in wrong module | Low | Low |
| 43 | group_rank.py | Opaque 4-tuple return type | Low | Low |
| 50 | prompts.py | Prompt says "integers only" for prices | Low | Trivial |
| 55 | models.py | `tags` str/list asymmetry undocumented | Low | Trivial |
| 56 | models.py | No `ingested_at` timestamp | Low — future scope | Low |
| 57 | tags.py | Comma-only separator assumption undocumented | Low | Trivial |
| 60 | sqlite_store.py | `fetchall()` loads full corpus into memory | Low | Low |
| 61 | sqlite_store.py | `date_snippets` not in SQL schema | Low — future scope | Low |
| 64 | api/main.py | Explanation errors silently swallowed | Low | Trivial |
| 65 | api/main.py | Opaque path rejection error | Low | Trivial |
| 66 | eval/metrics.py | Missing MRR and NDCG | Low — future scope | Medium |
| 16 | attributes.py | `"open air"` vs `"open-air"` | Low | Trivial |
| 17 | rules.py | `.lower()` called twice | Low | Trivial |
| 18 | attributes.py | `paid_language_hint` redundancy | Low | Low |
| 6 | event_attributes.py | Dead `"medium"` in `NoiseHint` | Low | Trivial |
| 10 | event_attributes.py | Free+paid conflict no validator | Low | Low |
| 4 | heuristics.py | No positive signal | Low — coarse filter by design | Medium |
| 12 | event_attributes.py | Missing fields | Low — future scope | Medium |
| 22 | text.py | Incomplete docstring | Info only | Trivial |
| 5 | heuristics.py | `combined` vs `raw_lower` note | Info only | None |
