"""FastAPI REST surface over retrieve + rank (local dev; add auth before public deploy)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException

log = logging.getLogger(__name__)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from belgrade_recommender.api.paths import repo_root, resolve_under_repo
from belgrade_recommender.llm.gemini_explain import (
    explain_ranking_plain,
    ranked_lines_from_group,
    ranked_lines_from_single,
)
from belgrade_recommender.rank.group_rank import (
    combined_group_query_text,
    fixture_user_to_prefs,
    rank_by_least_misery,
)
from belgrade_recommender.retrieve.service import (
    load_normalized_events,
    retrieve_from_jsonl_file,
    retrieve_from_jsonl_with_hard,
    retrieve_semantic_then_group_hard_filter,
)
from belgrade_recommender.schemas.parsed_preferences import ParsedHardConstraints


class SingleRecommendBody(BaseModel):
    events_path: str = Field(..., description="JSONL path relative to repo root")
    fixture_path: str = Field(default="fixtures/synthetic_users.json")
    fixture_user_id: str
    use_hard: bool = True
    top_k: int = Field(10, ge=1, le=50)
    semantic_pool: int | None = Field(
        default=None,
        ge=1,
        le=500,
        description="When use_hard: pool size (default max(60, top_k*5))",
    )
    faiss_index_dir: str | None = Field(default=None, description="Optional FAISS dir under repo root")
    include_explanation: bool = Field(
        default=False,
        description="If true, call Gemini after ranking (requires .[gemini] and API key).",
    )


class SingleHit(BaseModel):
    event_id: str
    score: float
    link: str
    title_preview: str


class SingleRecommendResponse(BaseModel):
    results: list[SingleHit]
    explanation: str | None = Field(
        default=None,
        description="Plain-language rationale when include_explanation was true.",
    )


class GroupRecommendBody(BaseModel):
    events_path: str
    fixture_path: str = Field(default="fixtures/synthetic_users.json")
    group_id: str
    semantic_pool: int = Field(80, ge=10, le=300)
    after_hard_k: int = Field(25, ge=1, le=80)
    top_n: int = Field(10, ge=1, le=30)
    faiss_index_dir: str | None = None
    include_explanation: bool = Field(
        default=False,
        description="If true, call Gemini after ranking (requires .[gemini] and API key).",
    )


class GroupHit(BaseModel):
    event_id: str
    semantic_score: float
    min_soft: float
    mean_soft: float
    link: str
    title_preview: str


class GroupRecommendResponse(BaseModel):
    results: list[GroupHit]
    explanation: str | None = None


app = FastAPI(
    title="Belgrade group event recommender",
    version="0.1.0",
    description="Local REST API. Run from repo root or set BELGRADE_RECOMMENDER_ROOT.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8501", "http://localhost:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/recommendations/single", response_model=SingleRecommendResponse)
def recommend_single(body: SingleRecommendBody) -> SingleRecommendResponse:
    log.info(
        "request [single] user=%s top_k=%d use_hard=%s events=%s",
        body.fixture_user_id, body.top_k, body.use_hard, body.events_path,
    )
    root = repo_root()
    try:
        events_file = resolve_under_repo(root, body.events_path, must_be_file=True, must_be_dir=False)
        fixture_file = resolve_under_repo(root, body.fixture_path, must_be_file=True, must_be_dir=False)
        faiss_dir: Path | None = None
        if body.faiss_index_dir:
            faiss_dir = resolve_under_repo(
                root,
                body.faiss_index_dir,
                must_be_file=False,
                must_be_dir=True,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid path") from exc

    data = json.loads(fixture_file.read_text(encoding="utf-8"))
    users = {u["user_id"]: u for u in data.get("users", [])}
    u = users.get(body.fixture_user_id)
    if not u:
        raise HTTPException(status_code=400, detail="Unknown fixture_user_id")

    query = u["preferences_plain_text"]
    pool = body.semantic_pool if body.semantic_pool is not None else max(60, body.top_k * 5)

    try:
        if body.use_hard:
            hard = ParsedHardConstraints.model_validate(u["hard_constraints"])
            ranked = retrieve_from_jsonl_with_hard(
                events_file,
                query,
                hard,
                semantic_pool=pool,
                result_k=body.top_k,
                faiss_index_dir=faiss_dir,
            )
        else:
            ranked = retrieve_from_jsonl_file(
                events_file,
                query,
                top_k=body.top_k,
                faiss_index_dir=faiss_dir,
            )
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="Retrieval dependencies not installed") from exc

    hits = [
        SingleHit(
            event_id=ev.event_id,
            score=float(score),
            link=ev.link,
            title_preview=(ev.event_description_resolved or "")[:240].replace("\n", " "),
        )
        for ev, score in ranked
    ]
    log.info("response [single] user=%s n_results=%d", body.fixture_user_id, len(hits))

    explanation: str | None = None
    if body.include_explanation:
        try:
            prefs = fixture_user_to_prefs(u)
            lines = ranked_lines_from_single(ranked)
            explanation = explain_ranking_plain(
                preferences_context=prefs.model_dump_json(),
                ranked_lines=lines,
            )
        except (ImportError, OSError, RuntimeError, ValueError):
            log.warning("explanation failed for user=%s", body.fixture_user_id, exc_info=True)
            explanation = None
    return SingleRecommendResponse(results=hits, explanation=explanation)


@app.post("/v1/recommendations/group", response_model=GroupRecommendResponse)
def recommend_group(body: GroupRecommendBody) -> GroupRecommendResponse:
    log.info(
        "request [group] group=%s top_n=%d pool=%s events=%s",
        body.group_id, body.top_n, body.semantic_pool, body.events_path,
    )
    root = repo_root()
    try:
        events_file = resolve_under_repo(root, body.events_path, must_be_file=True, must_be_dir=False)
        fixture_file = resolve_under_repo(root, body.fixture_path, must_be_file=True, must_be_dir=False)
        faiss_dir: Path | None = None
        if body.faiss_index_dir:
            faiss_dir = resolve_under_repo(
                root,
                body.faiss_index_dir,
                must_be_file=False,
                must_be_dir=True,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid path") from exc

    data = json.loads(fixture_file.read_text(encoding="utf-8"))
    users_by_id = {u["user_id"]: u for u in data.get("users", [])}
    group = next((g for g in data.get("groups", []) if g["group_id"] == body.group_id), None)
    if not group:
        raise HTTPException(status_code=400, detail="Unknown group_id")

    member_users: list[dict] = []
    for uid in group["member_user_ids"]:
        u = users_by_id.get(uid)
        if u is None:
            raise HTTPException(status_code=400, detail="Group references missing user")
        member_users.append(u)

    member_prefs = [fixture_user_to_prefs(u) for u in member_users]
    hards = [ParsedHardConstraints.model_validate(u["hard_constraints"]) for u in member_users]
    query = combined_group_query_text(member_users)

    try:
        events = load_normalized_events(events_file)
        candidates = retrieve_semantic_then_group_hard_filter(
            events,
            query,
            hards,
            semantic_pool=body.semantic_pool,
            result_k=body.after_hard_k,
            faiss_index_dir=faiss_dir,
        )
        ranked = rank_by_least_misery(candidates, member_prefs)[: body.top_n]
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="Retrieval dependencies not installed") from exc

    hits = [
        GroupHit(
            event_id=ev.event_id,
            semantic_score=float(sem),
            min_soft=float(mn),
            mean_soft=float(mean),
            link=ev.link,
            title_preview=(ev.event_description_resolved or "")[:240].replace("\n", " "),
        )
        for ev, sem, mn, mean in ranked
    ]
    log.info(
        "response [group] group=%s n_results=%d top_min_soft=%.3f",
        body.group_id, len(hits), ranked[0][2] if ranked else 0.0,
    )

    explanation: str | None = None
    if body.include_explanation:
        try:
            ctx = json.dumps(
                {
                    "group_id": body.group_id,
                    "members": [p.model_dump() for p in member_prefs],
                },
                ensure_ascii=False,
            )
            lines = ranked_lines_from_group(ranked)
            explanation = explain_ranking_plain(
                preferences_context=ctx,
                ranked_lines=lines,
            )
        except (ImportError, OSError, RuntimeError, ValueError):
            log.warning("explanation failed for group=%s", body.group_id, exc_info=True)
            explanation = None
    return GroupRecommendResponse(results=hits, explanation=explanation)
