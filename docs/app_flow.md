# App Flow & Retrieve Phase Deep-Dive

---

## Part 1 — What happens when you click a button

There are two buttons. Here is the full path for each.

---

### Button 1: "Pokreni retrieval" (single user, Tab 1)

```
┌─────────────────────────────────────────────────────────────────┐
│  STREAMLIT UI (streamlit_app.py)                                │
│                                                                 │
│  User picks a fixture user → query = preferences_plain_text    │
│  User sets: top_k, use_hard, optional FAISS dir                 │
│  → clicks "Pokreni retrieval"                                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  LOAD CORPUS  (retrieve/service.py)                             │
│                                                                 │
│  load_normalized_events(events_path)                            │
│    ├── if .db/.sqlite  → sqlite_store.load_all_normalized_events│
│    └── if .jsonl       → load_normalized_events_jsonl           │
│         reads line by line → NormalizedEvent.model_validate_json│
│                                                                 │
│  Result: list[NormalizedEvent]  (all events in memory)          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  SEMANTIC RETRIEVAL  (retrieve/service.py)                      │
│                                                                 │
│  pool = max(60, top_k × 5)    ← wider than top_k               │
│                                                                 │
│  Path A — no FAISS (in-memory NumPy):                           │
│    retrieve_top_k(events, query_text, top_k=pool)               │
│      1. For every event → build_embedding_text()  ─────────┐   │
│      2. encoder.encode(all_texts)  → corpus matrix (n × d)  │   │
│      3. encoder.encode([query])    → query vector (d,)       │   │
│      4. cosine_top_k(corpus, query_vec, pool)               │   │
│         → sorted list of (row_index, score)                 │   │
│                                                              │   │
│  Path B — FAISS (pre-built index):                          │   │
│    retrieve_top_k_faiss(events_by_id, index_dir, query)     │   │
│      1. encoder.encode([query])    → query vector (d,)       │   │
│      2. faiss_index.search(index_dir, query_vec, pool)      │   │
│         → (row_index, score) pairs from on-disk index       │   │
│      3. map row_index → event_id → NormalizedEvent          │   │
│                                                              │   │
│  Result: list of (NormalizedEvent, score) — up to `pool`    │   │
└────────────────────────┬─────────────────────────────────────   │
                         │                                        │
              [if use_hard=True]                                  │
                         ▼                                        │
┌─────────────────────────────────────────────────────────────────┐
│  HARD FILTER  (retrieve/hard_filters.py)                        │
│                                                                 │
│  For each (event, score) in semantic pool:                      │
│    attrs = extract_event_attributes_maybe_lazy_gemini(event)    │
│      ├── rules: price, city, dates, type, noise, outdoor        │
│      └── if USE_LAZY_GEMINI=1 and noise==unknown → Gemini call  │
│                                                                 │
│    passes_hard_constraints(hard, event, attrs)?                 │
│      ├── _city_ok()              city preference vs city_hint   │
│      ├── _free_and_budget_ok()   price signals vs budget cap    │
│      └── _forbidden_categories_ok() tag/hint vs forbidden list  │
│                                                                 │
│  Collect until result_k events pass → stop early               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  DISPLAY  (streamlit_app.py)                                    │
│                                                                 │
│  For each (event, score):                                       │
│    st.expander(f"{rank}. score={score:.4f} — {title[:80]}…")   │
│      → link + full title preview                                │
└─────────────────────────────────────────────────────────────────┘
```

---

### Button 2: "Pokreni grupni pipeline" (group, Tab 2)

```
┌─────────────────────────────────────────────────────────────────┐
│  STREAMLIT UI                                                   │
│                                                                 │
│  User picks a group → resolves member_user_ids                  │
│  User sets: semantic_pool, after_hard_k, show_top               │
│  → clicks "Pokreni grupni pipeline"                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  BUILD GROUP QUERY  (rank/group_rank.py)                        │
│                                                                 │
│  combined_group_query_text(member_users)                        │
│    → concatenate each member's preferences_plain_text           │
│    → single long query string representing the whole group      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  LOAD CORPUS  (same as single user flow above)                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  SEMANTIC RETRIEVAL + GROUP HARD FILTER                         │
│  (retrieve/service.py → retrieve_semantic_then_group_hard_filter)│
│                                                                 │
│  Same semantic search as above (pool = semantic_pool param)     │
│                                                                 │
│  For each (event, score) in pool:                               │
│    attrs = extract_event_attributes_maybe_lazy_gemini(event)    │
│    passes_all_hard_constraints(hards, event, attrs)             │
│      → ALL members must pass — intersection logic               │
│      → one member's hard veto removes the event for the group   │
│                                                                 │
│  Collect until after_hard_k events pass                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  SOFT SCORING + LEAST-MISERY RANK  (rank/group_rank.py)         │
│                                                                 │
│  rank_by_least_misery(candidates, member_prefs)                 │
│                                                                 │
│  For each candidate event:                                      │
│    per_user_soft_scores(members, event, semantic_sim)           │
│      For each member:                                           │
│        attrs = extract_event_attributes_maybe_lazy_gemini(event)│
│        soft_adjusted_score(prefs, event, attrs, semantic_sim)   │
│          base  = clamp(semantic_sim, 0, 1)                      │
│          bonus = +0.06 per liked_type match (cap 0.28)          │
│          penalty = -0.12 if noise mismatch (high vs low pref)   │
│          score = clamp(base + bonus - penalty, 0, 1)            │
│                                                                 │
│    min_soft  = min of all member scores  ← "least misery"       │
│    mean_soft = mean of all member scores                        │
│                                                                 │
│  Sort by (min_soft DESC, mean_soft DESC)                        │
│  Take top show_top                                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  DISPLAY                                                        │
│                                                                 │
│  For each (event, sem, min_soft, mean_soft):                    │
│    st.expander with all three scores + link + title             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 2 — The Retrieve Phase: Vector Search Explained

### What is a vector / embedding?

The sentence-transformer model takes a piece of text and converts it into a list of numbers — a **vector** (also called an embedding). For example `"all-MiniLM-L6-v2"` produces a vector of **384 numbers** for any input text, regardless of how long or short it is.

The key property: **texts with similar meaning produce vectors that point in a similar direction in 384-dimensional space.** "Jazz concert tonight" and "Live music event" will be closer to each other than "Jazz concert tonight" and "Preheat the oven to 200°C".

```
Text: "Jazz concert tonight at KC Grad"
         │
         ▼
  SentenceTransformer
  (all-MiniLM-L6-v2)
         │
         ▼
  [0.021, -0.134, 0.089, 0.412, -0.003, ... ]   ← 384 numbers
                      vector (d=384)
```

---

### Step 1 — Building the event text (`retrieve/text.py`)

Before encoding, each event is converted to a single string:

```python
def build_embedding_text(event, max_chars=2500):
    parts = [
        f"channel: {event.source_channel}",   # e.g. "channel: belgrade_events"
        f"tags: {', '.join(event.tags)}",      # e.g. "tags: concert, jazz, livemusic"
        f"description: {body}",                # the resolved English description
    ]
    return "\n".join(parts)[:2500]
```

This string is what the model actually sees. The channel and tags are prepended so the vector reflects structured metadata, not just the free text.

---

### Step 2 — Encoding (`retrieve/encoder.py`)

```python
class SentenceEncoder:
    def encode(self, texts: list[str]) -> np.ndarray:
        # loads model on first call (lazy)
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,   # ← important
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
```

`normalize_embeddings=True` means every output vector is **L2-normalized** — scaled so its length equals exactly 1. This is crucial because it makes the math simpler: for two unit vectors, the **dot product equals cosine similarity**.

```
Without normalization:  cosine(a, b) = (a · b) / (|a| × |b|)
With normalization:     cosine(a, b) = a · b        (since |a|=|b|=1)
```

So the model returns a matrix of shape `(n_events, 384)` where every row is a unit vector.

---

### Step 3 — Similarity Search

There are two modes depending on whether a FAISS index is available:

---

#### Mode A — In-memory NumPy (`retrieve/cosine_search.py`)

```
corpus_matrix: shape (n_events, 384)
query_vector:  shape (384,)

scores = corpus_matrix @ query_vector   ← one dot product per event
                                           result: shape (n_events,)

order  = argsort(-scores)[:pool]        ← sort descending, take top pool
```

This is a single matrix-vector multiply — very fast in NumPy even for thousands of events. The result is a score in `[-1, 1]` for each event: `1.0` means perfectly aligned direction (identical meaning), `0.0` means orthogonal (unrelated), `-1.0` means opposite.

```
Example scores:
  event_001 "Jazz evening at Bitef"     → 0.82   ← top hit
  event_002 "Techno night at Drugstore" → 0.21
  event_003 "Food festival Kalemegdan"  → 0.14
  event_004 "Dobro jutro beograde..."   → 0.03
```

---

#### Mode B — FAISS index (`retrieve/faiss_index.py`)

For larger corpora, building the index once offline and loading it at query time is much faster than encoding all events on every request.

**Build time** (offline, `scripts/build_vector_index.py`):
```
All events → build_embedding_text → encode → (n, 384) matrix
→ faiss.IndexFlatIP(384)     ← IP = inner product (= cosine for normalized)
→ index.add(vectors)
→ write to disk:
     events.faiss       ← the index
     event_ids.json     ← maps row number → event_id
     manifest.json      ← records model name and dimensions
```

**Query time** (every request):
```
query text → encode → query_vector (384,)
→ load events.faiss from disk
→ index.search(query_vector, k=pool)
   returns: (scores, row_indices) — top pool hits
→ row_indices → event_ids.json → NormalizedEvent objects
```

`IndexFlatIP` does **exact** nearest-neighbour search (not approximate) — it checks every vector in the index but does it in optimized C++ with SIMD instructions, so it is much faster than the NumPy loop for large corpora.

---

### Step 4 — The Semantic Pool concept

Neither search mode returns the final top_k directly. They return a **pool** that is intentionally larger:

```
top_k = 10  (what the user wants)
pool  = max(60, 10 × 5) = 60

Semantic search returns 60 events sorted by vector similarity
         │
         ▼  hard filter loop
  event 1  → passes hard constraints → add to results (1/10)
  event 2  → FAILS city constraint   → skip
  event 3  → passes                  → add to results (2/10)
  event 4  → FAILS must_be_free      → skip
  event 5  → passes                  → add to results (3/10)
  ...
  event 23 → passes                  → add to results (10/10) ← stop
```

Why this matters: if you only retrieved 10 events semantically and some fail hard filters, you end up with fewer than 10 results. The pool ensures there are enough candidates to survive filtering and still fill the requested top_k.

---

### Full data shape summary

```
                        shape           dtype    description
                        ──────────────  ───────  ──────────────────────
corpus_matrix           (n_events, 384) float32  one row per event
query_vector            (384,)          float32  the user query
scores                  (n_events,)     float32  cosine similarity [-1, 1]
top_pool_indices        (pool,)         int      row numbers of top pool
```

---

### Why `all-MiniLM-L6-v2` and its limitation

This model is:
- Fast: only 6 transformer layers ("L6"), 22M parameters
- Good general English semantic similarity
- Max input: 256 word-pieces (text beyond that is silently truncated by the model itself)

Its limitation for this project: it is **English-only**. Serbian and Russian event descriptions will produce lower-quality vectors because the model was never trained on those languages. This is why issue #32 in the improvements doc recommends switching to `"paraphrase-multilingual-MiniLM-L12-v2"`, which was trained on 50+ languages including Serbian and Russian.
