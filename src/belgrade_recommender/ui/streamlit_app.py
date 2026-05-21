"""Lokalni Streamlit demo: retrieval i grupno rangiranje nad JSONL korpusom."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _path_under_repo(raw: str, *, is_file: bool, is_dir: bool) -> Path | None:
    raw = raw.strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (_repo_root() / p).resolve()
    else:
        p = p.resolve()
    root = _repo_root().resolve()
    try:
        if not p.is_relative_to(root):
            st.error("Put mora biti unutar foldera projekta (bez .. izvan repoa).")
            return None
    except ValueError:
        st.error("Nevalidna putanja.")
        return None
    if is_file and not p.is_file():
        st.error("JSONL fajl nije pronađen na toj putanji.")
        return None
    if is_dir and not p.is_dir():
        st.error("FAISS folder nije pronađen.")
        return None
    return p


@st.cache_data
def _load_fixture_bundle(path_str: str) -> tuple[list[dict], list[dict]]:
    p = Path(path_str)
    data = json.loads(p.read_text(encoding="utf-8"))
    return data.get("users", []), data.get("groups", [])


def main() -> None:
    st.set_page_config(page_title="Belgrade recommender demo", layout="wide")
    st.title("Belgrade group event recommender — demo")
    st.caption(
        "Ručna provera kvaliteta: nema automatske metrike relevantnosti — "
        "proceni da li naslovi i linkovi imaju smisla za upit/grupu.",
    )

    root = _repo_root()
    default_events = root / "fixtures" / "events_smoke.jsonl"

    events_path_in = st.text_input(
        "JSONL događaja (normalized)",
        value=str(default_events.relative_to(root)) if default_events.is_file() else "fixtures/events_smoke.jsonl",
        help="Relativna putanja od korena projekta ili apsolutna unutar repoa.",
    )
    events_path = _path_under_repo(events_path_in, is_file=True, is_dir=False)
    if events_path is None:
        st.stop()

    faiss_in = st.text_input(
        "FAISS indeks (opciono, prazno = u memoriji / NumPy)",
        value="",
        help="Folder sa events.faiss + event_ids.json + manifest.json",
    )
    faiss_dir = None
    if faiss_in.strip():
        faiss_dir = _path_under_repo(faiss_in, is_file=False, is_dir=True)
        if faiss_dir is None:
            st.stop()

    fixture_path_in = st.text_input(
        "Fixture korisnika / grupa",
        value="fixtures/synthetic_users.json",
    )
    fp = _path_under_repo(fixture_path_in, is_file=True, is_dir=False)
    if fp is None:
        st.stop()
    users, groups = _load_fixture_bundle(str(fp))

    tab1, tab2 = st.tabs(["Jedan korisnik (retrieval)", "Grupa (hard + rang)"])

    with tab1:
        user_ids = [u["user_id"] for u in users]
        choice = st.selectbox("Korisnik iz fixture-a", user_ids)
        use_hard = st.checkbox("Primeni hard_constraints", value=True)
        top_k = st.slider("Broj rezultata", 1, 30, 10, key="sk")

        if st.button("Pokreni retrieval", key="run_single"):
            u = next(x for x in users if x["user_id"] == choice)
            query = u["preferences_plain_text"]
            with st.spinner("Embedding + pretraga…"):
                try:
                    if use_hard:
                        hard = ParsedHardConstraints.model_validate(u["hard_constraints"])
                        ranked = retrieve_from_jsonl_with_hard(
                            events_path,
                            query,
                            hard,
                            semantic_pool=max(60, top_k * 5),
                            result_k=top_k,
                            faiss_index_dir=faiss_dir,
                        )
                    else:
                        ranked = retrieve_from_jsonl_file(
                            events_path,
                            query,
                            top_k=top_k,
                            faiss_index_dir=faiss_dir,
                        )
                except ImportError as e:
                    st.error(str(e))
                    st.stop()

            st.subheader("Rezultati")
            for i, (ev, score) in enumerate(ranked, start=1):
                title = (ev.event_description_resolved or "")[:200].replace("\n", " ")
                with st.expander(f"{i}. score={score:.4f} — {title[:80]}…"):
                    st.write(ev.link)
                    st.text(title)

    with tab2:
        gids = [g["group_id"] for g in groups]
        gid = st.selectbox("Grupa", gids, key="grp")
        pool = st.slider("Semantički pool", 20, 120, 80, key="pool")
        after_hard = st.slider("Max kandidata posle hard filtera", 5, 40, 25, key="ah")
        show_top = st.slider("Prikaži top N posle rangiranja", 1, 20, 10, key="topn")

        if st.button("Pokreni grupni pipeline", key="run_group"):
            group = next(g for g in groups if g["group_id"] == gid)
            users_by_id = {u["user_id"]: u for u in users}
            member_users = [users_by_id[uid] for uid in group["member_user_ids"]]
            member_prefs = [fixture_user_to_prefs(u) for u in member_users]
            hards = [ParsedHardConstraints.model_validate(u["hard_constraints"]) for u in member_users]
            query = combined_group_query_text(member_users)

            with st.spinner("Grupni retrieval + rang…"):
                try:
                    events = load_normalized_events(events_path)
                    candidates = retrieve_semantic_then_group_hard_filter(
                        events,
                        query,
                        hards,
                        semantic_pool=pool,
                        result_k=after_hard,
                        faiss_index_dir=faiss_dir,
                    )
                    ranked = rank_by_least_misery(candidates, member_prefs)[:show_top]
                except ImportError as e:
                    st.error(str(e))
                    st.stop()

            st.subheader("Rang (least-misery)")
            for i, (ev, sem, mn, mean) in enumerate(ranked, start=1):
                title = (ev.event_description_resolved or "")[:200].replace("\n", " ")
                with st.expander(
                    f"{i}. sem={sem:.4f} min_soft={mn:.4f} mean={mean:.4f} — {title[:70]}…",
                ):
                    st.write(ev.link)
                    st.text(title)


if __name__ == "__main__":
    main()
