# Belgrade Group Event Recommender

A group-aware Telegram bot that finds Belgrade events everyone in your chat can enjoy.
Each member describes their preferences in plain text — the system parses them, searches a corpus of thousands of real events, and returns the best matches using a least-misery group ranking (no one gets left out).

> **Deployment:** The bot is deployed to the cloud and ready to use — jump straight to [Using the Bot](#using-the-bot--user-guide) if you just want to try it.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Application Components](#application-components)
- [Developer Guide](#developer-guide)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
  - [Running the Bot Locally](#running-the-bot-locally)
  - [Running the REST API](#running-the-rest-api)
  - [Running Tests](#running-tests)
  - [Docker](#docker)
- [Using the Bot — User Guide](#using-the-bot--user-guide)
- [Cloud Deployment](#cloud-deployment)
- [Tech Stack](#tech-stack)
- [Evaluation Results](#evaluation-results)
- [Troubleshooting](#troubleshooting)

---

## How It Works

```
User messages (plain text)
        │
        ▼
┌─────────────────────┐
│  Preference Parser  │  OpenAI extracts structured hard + soft preferences
│  (LLM / OpenAI)     │  from each member's free-text input
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Semantic Retrieval │  Sentence-transformer encodes the group query;
│  + Keyword Fallback │  cosine top-80 from corpus; keyword search if
│                     │  top score < 0.30
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Hard Filtering    │  Removes events that violate ANY member's
│                     │  hard constraints (city, budget, forbidden)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Group Ranking      │  Least-misery: sort by (min member score,
│  (Least-Misery)     │  then mean score) — worst-off member decides
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  LLM Explanation    │  One OpenAI call generates a short reason
│                     │  per event in the user's language
└────────┬────────────┘
         │
         ▼
   Telegram reply with ranked results + conflict warnings
```

---

## Application Components

### `src/belgrade_recommender/`

| Module | Role |
|---|---|
| `bot/` | Telegram bot — handlers, session management, pipeline orchestration |
| `ingest/` | Loads raw JSONL, resolves placeholders, assigns tags, produces `events_normalized.jsonl` |
| `extract/` | Rule-based attribute extraction: price, city, noise level, event type hints |
| `retrieve/` | Sentence-transformer encoder, cosine search, FAISS index, hard filters, keyword fallback |
| `rank/` | Per-user soft scoring and group aggregation (least-misery) |
| `llm/` | OpenAI wrappers — preference parsing and per-event explanations |
| `schemas/` | Pydantic models for preferences, event attributes, and parsed outputs |
| `eval/` | Offline evaluation — P@k, R@k, MRR, NDCG, Jain's fairness index |
| `api/` | FastAPI REST service (alternative to the bot for programmatic access) |
| `ui/` | Streamlit demo app for local manual testing |
| `storage/` | SQLite layer for normalized events |

### Key Files

| File | What it does |
|---|---|
| `bot/pipeline.py` | Full recommendation pipeline — called once per `/find` request |
| `bot/handlers.py` | Telegram command and message handlers |
| `bot/session.py` | Collects member messages during the 90-second window |
| `retrieve/cosine_search.py` | Dot-product top-k on L2-normalized embeddings |
| `retrieve/keyword_search.py` | Token-overlap fallback when semantic similarity is too low |
| `rank/group_rank.py` | Least-misery aggregation + monthly deduplication |
| `rank/soft_score.py` | Per-user soft score: semantic base + liked-type bonuses + noise penalties |
| `llm/openai_preferences.py` | Parses free-text into structured `ParsedUserPreferences` |
| `llm/openai_explain.py` | Generates short event explanations in the user's language |

---

## Developer Guide

### Prerequisites

- Python 3.11 or 3.12
- A Telegram bot token (create one via [@BotFather](https://t.me/BotFather))
- An OpenAI API key
- The normalized event corpus (`events_normalized_test.jsonl`) in `data/processed/`

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/naleeeee16/Belgrade-group-recommender.git
cd Belgrade-group-recommender

# 2. Create a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install the package with all dependencies
pip install -e ".[dev,retrieve,bot,api]"
```

### Environment Variables

Create a `.env` file in the project root:

```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
OPENAI_API_KEY=your_openai_api_key

# Optional
GEMINI_API_KEY=your_gemini_api_key
SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2
USE_LAZY_GEMINI_EVENT_ATTRIBUTES=0
```

> The `.env` file is gitignored — never commit API keys.

### Running the Bot Locally

```bash
# Make sure the corpus file exists at:
# data/processed/events_normalized_test.jsonl

python run_bot.py
```

On first run the sentence-transformer model downloads from Hugging Face (~90 MB). Subsequent starts are fast.

Expected startup output:
```
INFO  bot.pipeline  Loading corpus from data/processed/events_normalized_test.jsonl …
INFO  bot.pipeline  4821 indexable events loaded
INFO  bot.pipeline  Corpus encoded in 12.3s — shape (4821, 384)  model=all-MiniLM-L6-v2
INFO  bot.main      Bot is running...
```

### Running the REST API

```bash
# Development server with auto-reload
make api

# or manually:
uvicorn belgrade_recommender.api.main:app --reload --host 127.0.0.1 --port 8000
```

Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Example request:
```bash
curl -X POST http://127.0.0.1:8000/v1/recommendations/group \
  -H "Content-Type: application/json" \
  -d '{
    "events_path": "fixtures/events_smoke.jsonl",
    "user_texts": {
      "1": "Volim jazz i besplatne koncerte, bez techno žurki",
      "2": "Outdoor aktivnosti, priroda, vikend"
    },
    "top_k": 5
  }'
```

### Running Tests

```bash
pytest -q
```

All external APIs (OpenAI, Gemini, Telegram) are mocked — tests run offline in under 10 seconds.

To run the offline evaluation (requires corpus file):
```bash
python scripts/run_eval.py
```

### Docker

Builds and starts the FastAPI service only (the bot runs separately):

```bash
docker compose build api
docker compose up api
```

Swagger UI available at [http://localhost:8000/docs](http://localhost:8000/docs).

To include OpenAI-powered explanations in Docker, set `OPENAI_API_KEY` in `docker-compose.yml` or pass it as an environment variable:

```bash
OPENAI_API_KEY=sk-... docker compose up api
```

---

## Project Structure

```
belgrade_group_recommender/
├── src/
│   └── belgrade_recommender/
│       ├── bot/
│       │   ├── pipeline.py        # full recommendation pipeline
│       │   ├── handlers.py        # Telegram command + message handlers
│       │   ├── session.py         # per-chat preference collection window
│       │   ├── formatter.py       # result formatting + conflict messages
│       │   ├── config.py          # shared constants (score floor, timeouts)
│       │   └── main.py            # bot startup + corpus loading
│       ├── ingest/
│       │   ├── pipeline.py        # JSONL → NormalizedEvent
│       │   ├── heuristics.py      # include_in_index classification
│       │   ├── models.py          # NormalizedEvent Pydantic model
│       │   └── tags.py            # tag normalization helpers
│       ├── extract/
│       │   ├── attributes.py      # EventAttributes extraction entry point
│       │   └── rules.py           # regex rules: price, city, noise, type hints
│       ├── retrieve/
│       │   ├── cosine_search.py   # dot-product top-k on normalized embeddings
│       │   ├── keyword_search.py  # token-overlap fallback (low cosine scores)
│       │   ├── encoder.py         # SentenceEncoder wrapper
│       │   ├── hard_filters.py    # per-constraint and group-intersection checks
│       │   ├── faiss_index.py     # optional on-disk FAISS index
│       │   ├── text.py            # build_embedding_text per event
│       │   └── service.py         # corpus loaders (JSONL, SQLite, FAISS)
│       ├── rank/
│       │   ├── group_rank.py      # least-misery aggregation + deduplication
│       │   └── soft_score.py      # per-user semantic + liked-type + noise score
│       ├── llm/
│       │   ├── openai_preferences.py  # preference parsing via OpenAI
│       │   ├── openai_explain.py      # per-event explanation generation
│       │   └── prompts.py             # all prompt templates
│       ├── schemas/
│       │   ├── parsed_preferences.py  # ParsedUserPreferences, Hard/Soft constraints
│       │   └── event_attributes.py    # EventAttributes schema
│       ├── eval/
│       │   └── metrics.py         # P@k, R@k, MRR, NDCG, fairness metrics
│       ├── api/
│       │   ├── main.py            # FastAPI app + CORS
│       │   └── paths.py           # path validation (directory traversal guard)
│       ├── storage/
│       │   └── sqlite_store.py    # SQLite schema + queries
│       └── ui/
│           └── streamlit_app.py   # local Streamlit demo
├── scripts/
│   ├── run_eval.py                # offline evaluation runner
│   ├── run_ingest.py              # ingest raw JSONL → normalized
│   └── ...                        # additional demo + audit scripts
├── tests/                         # pytest unit + integration tests (20 files)
├── fixtures/
│   ├── eval_labels.json           # 15 labeled evaluation cases
│   └── synthetic_users.json       # 4 personas for manual testing
├── data/
│   ├── raw/                       # events_dataset.jsonl (gitignored)
│   └── processed/                 # normalized JSONL + FAISS index (gitignored)
├── run_bot.py                     # bot entrypoint
├── pyproject.toml                 # package metadata + dependency extras
├── Dockerfile                     # FastAPI image
└── docker-compose.yml             # API service definition
```

---

## Using the Bot — User Guide

The bot is live on Telegram. You do not need to install anything.

### Step 1 — Add the Bot to Your Group

1. Open Telegram and search for **@BelgradeEventsBot** (or use the link your team shared)
2. Open the bot's profile and tap **Add to Group**
3. Select your group chat and confirm

> If you want to use the bot in a private chat (just you), you can also message it directly — it works for single users too.

### Step 2 — Start a Search Session

In your group chat, type:

```
/find
```

or the Serbian alias:

```
/preporuci
```

The bot will reply with instructions and open a **90-second window** for members to submit their preferences.

### Step 3 — Each Member Types Their Preferences

Every person in the group sends **one message** describing what they want. Write naturally — any language works (Serbian, English, Russian):

```
Hoću nešto besplatno, tiho, u centru. Ne volim techno.
```

```
Outdoor hiking or a park, doesn't matter if it costs something
```

```
Хочу джаз или живую музыку, желательно бесплатно
```

You can mention:
- **What you like:** jazz, hiking, art, food, comedy, cinema, theatre...
- **Budget:** free, under 500 RSD, no limit
- **Atmosphere:** quiet/loud, indoor/outdoor
- **Location:** Belgrade city center, Novi Sad, anywhere
- **What to avoid:** no techno, no clubs, no smoking areas

### Step 4 — Receive Recommendations

After 90 seconds (or once everyone has responded), the bot processes all preferences together and replies with the **top events** that work for the whole group:

```
🎵 Jazz Night at KC Grad
   Why: Free entry, quiet atmosphere, central location — matches all members
   🔗 t.me/...

🌿 Kalemegdan Park Photography Walk
   Why: Outdoor, free, no noise — great for the quiet preference
   🔗 t.me/...
```

If members have conflicting preferences (e.g., one wants loud music, another wants quiet), the bot will flag this upfront before showing results.

### Tips

| Tip | Details |
|---|---|
| Be specific | "free jazz" works better than "fun event" |
| Mention dealbreakers | "no smoking areas", "not after midnight" help the hard filter |
| One message per person | The bot collects one preference block per member |
| Works for 1–8 people | Groups larger than 8 may have very few events that satisfy everyone |

---

## Cloud Deployment

The bot and API are deployed to the cloud. No local setup is needed to use the bot.

### Architecture

```
Internet
   │
   ├── Telegram users ──► Telegram servers ──► Bot process (cloud VM)
   │                                                │
   │                                         pipeline.py runs
   │                                         OpenAI API calls
   │
   └── API clients ──► FastAPI (Docker container, cloud VM)
                              │
                         corpus in memory
                         (loaded at startup)
```

### Deployment Steps (for maintainers)

**1. Provision a cloud VM** (Railway, Render, GCP, DigitalOcean, etc.)

Required: 2 GB RAM minimum (embedding model + corpus), Python 3.11+ or Docker.

**2. Copy the corpus to the server**

```bash
scp data/processed/events_normalized_test.jsonl user@server:/app/data/processed/
```

**3. Set environment variables on the server**

```bash
export TELEGRAM_BOT_TOKEN=...
export OPENAI_API_KEY=...
```

**4. Start the API (Docker)**

```bash
docker compose build api
docker compose up -d api
```

**5. Start the bot**

```bash
pip install -e ".[retrieve,bot]"
nohup python run_bot.py > logs/bot.log 2>&1 &
```

**6. Verify**

```bash
curl http://localhost:8000/docs   # API health check
tail -f logs/bot.log              # bot startup logs
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Bot interface** | [python-telegram-bot](https://python-telegram-bot.org/) v21 (async) |
| **Preference parsing** | OpenAI GPT-4o-mini with structured output (Pydantic schema) |
| **Embeddings** | `sentence-transformers` — `all-MiniLM-L6-v2` |
| **Similarity search** | In-memory cosine (numpy) or on-disk FAISS |
| **Keyword fallback** | Custom token-overlap search (`retrieve/keyword_search.py`) |
| **Event explanations** | OpenAI GPT-4o-mini (one call per `/find` request) |
| **REST API** | FastAPI + Uvicorn |
| **Data validation** | Pydantic v2 |
| **Storage** | JSONL (primary corpus) + SQLite (optional relational layer) |
| **Containerization** | Docker + Docker Compose |
| **Testing** | pytest (20 test files, all external APIs mocked) |

---

## Evaluation Results

On a labeled test set of 15 cases across 4 user personas:

| Metric | Semantic only | Full pipeline |
|---|---|---|
| P@k | 0.112 | **0.148** |
| R@k | 0.492 | **0.628** |
| MRR | 0.501 | **0.584** |
| NDCG | 0.523 | **0.601** |

Hard-constraint filtering improves NDCG by +0.08 and MRR by +0.08. The first relevant event typically appears within the top 2–3 results (MRR ≈ 0.58).

---

## Troubleshooting

**Bot does not respond after `/find`**

The bot waits 90 seconds for member messages. If no one sends preferences before the window closes, it exits silently. Type `/find` again and send your preference message within 90 seconds.

**`FileNotFoundError: Normalized corpus not found`**

The corpus file is not in git (too large). Run the ingest pipeline or ask the maintainer for the file:
```bash
python scripts/run_ingest.py --input data/raw/events_dataset.jsonl \
  --output data/processed/events_normalized_test.jsonl
```

**First startup is slow**

The sentence-transformer model (~90 MB) downloads from Hugging Face on first run. This is normal. Subsequent starts use the local cache and take ~10 seconds.

**`openai.RateLimitError` in logs**

The bot falls back to semantic-only mode when OpenAI is rate-limited — results still appear but without structured preference parsing. Upgrade your OpenAI tier or reduce request frequency.

**`TELEGRAM_BOT_TOKEN is not set`**

The `.env` file is missing or not in the project root. Create it as shown in [Environment Variables](#environment-variables).

**Bot returns no results**

This usually means all candidates failed the hard filter. The most common cause: one member has a very strict constraint (e.g., `must_be_free` + specific city) that no event satisfies. Try relaxing constraints or check logs:
```bash
tail -f logs/bot.log | grep "Step 3"
```
