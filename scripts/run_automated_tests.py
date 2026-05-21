#!/usr/bin/env python3
"""
Automated pipeline test for the Belgrade group recommender.

Usage:
    python run_automated_tests.py
    python run_automated_tests.py --dataset fixtures/events_dataset.jsonl
    python run_automated_tests.py --skip-ingest   # reuse already-normalized file

Requirements:
    pip install -e ".[retrieve]"
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Windows PowerShell defaults to cp1252 — force UTF-8 so Serbian chars don't crash.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

# ---------------------------------------------------------------------------
# Path bootstrap — script lives at project root
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Load .env file if present (no external dependencies needed)
# ---------------------------------------------------------------------------
import os
_env_file = _ROOT / ".env"
if _env_file.is_file():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("autotest")

import numpy as np

from belgrade_recommender.extract.attributes import extract_event_attributes_maybe_lazy_gemini
from belgrade_recommender.ingest.models import NormalizedEvent
from belgrade_recommender.ingest.pipeline import ingest_jsonl_file
from belgrade_recommender.rank.group_rank import rank_by_least_misery
from belgrade_recommender.retrieve.cosine_search import cosine_top_k
from belgrade_recommender.retrieve.encoder import SentenceEncoder
from belgrade_recommender.retrieve.hard_filters import (
    passes_all_hard_constraints,
    passes_hard_constraints,
)
from belgrade_recommender.retrieve.service import load_normalized_events
from belgrade_recommender.retrieve.text import build_embedding_text
from belgrade_recommender.schemas.event_attributes import EventAttributes
from belgrade_recommender.schemas.parsed_preferences import (
    ParsedHardConstraints,
    ParsedSoftPreferences,
    ParsedUserPreferences,
)
from belgrade_recommender.bot.pipeline import _is_non_event


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _member(
    name: str,
    query: str,
    *,
    liked_types: list[str] | None = None,
    noise: str = "unknown",
    must_be_free: bool | None = None,
    forbidden: list[str] | None = None,
    budget_max_rsd: int | None = None,
    earliest_hour: int | None = None,
) -> dict:
    prefs = ParsedUserPreferences(
        summary_one_line=name,
        hard_constraints=ParsedHardConstraints(
            must_be_free=must_be_free,
            forbidden_event_categories=forbidden or [],
            budget_max_rsd=budget_max_rsd,
            earliest_start_hour_weekday=earliest_hour,
        ),
        soft_preferences=ParsedSoftPreferences(
            liked_types=liked_types or [],
            noise_tolerance=noise,  # type: ignore[arg-type]
        ),
    )
    return {"name": name, "query": query, "prefs": prefs}


# ---------------------------------------------------------------------------
# Test groups
# ---------------------------------------------------------------------------

def build_test_groups() -> list[dict]:
    return [

        # ------------------------------------------------------------------ #
        # G01 — Žurka vs. galerija                                            #
        # Luka hoće alkohol i provod, Marko hoće kulturu, Ana kompromis       #
        # ------------------------------------------------------------------ #
        {
            "group_id": "G01_zurka_vs_galerija",
            "description": "Luka hoće da pije, Marko hoće galeriju, Ana negde između",
            "members": [
                _member(
                    "Luka",
                    query="zelim da se dobro provedem i da pijem alkohol, neki bar ili zurka",
                    liked_types=["bar", "nightlife", "party", "concert"],
                    noise="high",
                    earliest_hour=20,
                ),
                _member(
                    "Marko",
                    query="zelim da idemo u galeriju ili neki kulturni dogadjaj, nesto mirno",
                    liked_types=["gallery", "exhibition", "museum", "culture"],
                    noise="low",
                    forbidden=["nightclub", "rave", "techno"],
                ),
                _member(
                    "Ana",
                    query="ok sam i sa zurkom i sa kulturom, bitno da je zabavno i da ne kostа mnogo",
                    liked_types=["concert", "festival", "outdoor", "bar"],
                    noise="medium",
                    budget_max_rsd=1500,
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # G02 — Studenti bez para                                             #
        # ------------------------------------------------------------------ #
        {
            "group_id": "G02_studenti_bez_para",
            "description": "Studenti — sve mora biti besplatno",
            "members": [
                _member(
                    "Tijana",
                    query="nemam para uopste, trazim nesto besplatno za vikend u Beogradu",
                    liked_types=["outdoor", "festival", "market", "social"],
                    noise="medium",
                    must_be_free=True,
                ),
                _member(
                    "Filip",
                    query="student sam, mogu samo na besplatne dogadjaje, volim muziku i umetnost",
                    liked_types=["concert", "art", "exhibition", "music"],
                    noise="medium_high",
                    must_be_free=True,
                ),
                _member(
                    "Jovana",
                    query="besplatno obavezno, preferam nesto napolju, park ili trg",
                    liked_types=["outdoor", "park", "open_air", "nature"],
                    noise="medium",
                    must_be_free=True,
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # G03 — Porodični izlet sa decom                                      #
        # ------------------------------------------------------------------ #
        {
            "group_id": "G03_porodicni_izlet",
            "description": "Porodica sa decom — sigurno, tiho, porodično okruženje",
            "members": [
                _member(
                    "Milica (mama)",
                    query="trazimo nesto za decu, porodicno, bezbedno i zabavno, ne preglasno",
                    liked_types=["family", "outdoor", "park", "workshop", "nature"],
                    noise="low",
                    forbidden=["nightclub", "rave", "techno", "bar"],
                ),
                _member(
                    "Dragan (tata)",
                    query="nesto aktivno za celu porodicu, priroda ili sport, deca da se istrce",
                    liked_types=["outdoor", "sports", "hiking", "nature", "park"],
                    noise="medium",
                    forbidden=["nightclub", "rave", "techno"],
                ),
                _member(
                    "Baka Vera",
                    query="nesto kulturno i mirno, muzej ili izlozba, da sednemo i odmorimo",
                    liked_types=["museum", "exhibition", "classical_music", "culture"],
                    noise="low",
                    forbidden=["nightclub", "rave", "techno", "concert"],
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # G04 — Devојачko veče                                                #
        # ------------------------------------------------------------------ #
        {
            "group_id": "G04_devojacko_vece",
            "description": "Devojačko veče — koktelи, ples, dobra atmosfera",
            "members": [
                _member(
                    "Maja",
                    query="organizujem devojacko vece, hocemo da plесamo i pijemo cocktaile, nesto glamurozno",
                    liked_types=["bar", "cocktail", "nightlife", "dance", "party"],
                    noise="high",
                    earliest_hour=21,
                ),
                _member(
                    "Katarina",
                    query="da, zurka je ok ali da nije preterano glasno, volim i dobre provode i razgovor",
                    liked_types=["bar", "lounge", "cocktail", "party"],
                    noise="medium_high",
                    earliest_hour=20,
                ),
                _member(
                    "Sofija",
                    query="sve mi je ok, jedino ne volim techno, hocemo nesto gde mozemo i da sedimo i pricamo",
                    liked_types=["bar", "lounge", "live_music", "social"],
                    noise="medium",
                    forbidden=["techno", "rave"],
                    earliest_hour=20,
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # G05 — Planinari i prirodnjaci                                       #
        # ------------------------------------------------------------------ #
        {
            "group_id": "G05_planinari",
            "description": "Planinari — izleti, priroda, aktivan provod",
            "members": [
                _member(
                    "Nemanja",
                    query="obozavam planinarenje i prirodu, trazim izlet ili turu oko Beograda za vikend",
                    liked_types=["hiking", "nature", "excursion", "outdoor"],
                    noise="low",
                ),
                _member(
                    "Isidora",
                    query="volim aktivne izlete, biciklizam ili pesacenje, bitan mi je svez vazduh",
                    liked_types=["cycling", "hiking", "outdoor", "nature", "sports"],
                    noise="low",
                ),
                _member(
                    "Rastko",
                    query="priroda je super ali i neki piknik ili roštilj uz to bio bi odlican",
                    liked_types=["outdoor", "nature", "picnic", "food", "hiking"],
                    noise="medium",
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # G06 — Džez i klasika                                                #
        # ------------------------------------------------------------------ #
        {
            "group_id": "G06_dzez_i_klasika",
            "description": "Ljubitelji muzike — džez fan vs. klasična muzika vs. world music",
            "members": [
                _member(
                    "Vladimir",
                    query="volim dzez, trazim dzez koncert ili bar sa dzez muzikom u Beogradu",
                    liked_types=["jazz", "live_music", "concert", "bar"],
                    noise="medium",
                    earliest_hour=19,
                ),
                _member(
                    "Biljana",
                    query="preferam klasicnu muziku ili operu, nesto ozbiljno i kvalitetno",
                    liked_types=["classical_music", "opera", "concert", "philharmonic"],
                    noise="low",
                    forbidden=["rave", "techno", "nightclub"],
                ),
                _member(
                    "Ognjen",
                    query="ok sam i sa dzezom i sa klasikom, bitno da je ziva muzika i dobra atmosfera",
                    liked_types=["live_music", "jazz", "world_music", "concert"],
                    noise="medium",
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # G07 — Foodie ekipa                                                  #
        # ------------------------------------------------------------------ #
        {
            "group_id": "G07_foodie_ekipa",
            "description": "Gastronomija — svako hoće drugačiju kuhinju",
            "members": [
                _member(
                    "Teodora",
                    query="obozavam azijsku kuhinju, sushi, ramen, trazim dobar azijski restoran ili food event",
                    liked_types=["food", "asian_food", "restaurant", "gastronomy"],
                    noise="medium",
                ),
                _member(
                    "Lazar",
                    query="volim domace srpsko jelo, rostilj, nesto tradicionalno i autenticno",
                    liked_types=["food", "traditional", "restaurant", "gastronomy"],
                    noise="medium",
                ),
                _member(
                    "Neda",
                    query="ja sam veganka, trazim vegan friendly restoran ili food festival sa opcijama za mene",
                    liked_types=["food", "vegan", "restaurant", "food_festival"],
                    noise="medium",
                    forbidden=["rave", "nightclub"],
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # G08 — Tech i startup scena                                          #
        # ------------------------------------------------------------------ #
        {
            "group_id": "G08_tech_startup",
            "description": "Tech ekipa — konferencije, meetupi, inovacije",
            "members": [
                _member(
                    "Stefan",
                    query="trazim tech meetup ili startup dogadjaj u Beogradu, networking je bitan",
                    liked_types=["tech", "startup", "lecture", "networking", "workshop"],
                    noise="low",
                ),
                _member(
                    "Aleksandra",
                    query="interesuju me radionice o vestackoj inteligenciji i novim tehnologijama",
                    liked_types=["ai", "tech", "workshop", "lecture", "innovation"],
                    noise="low",
                ),
                _member(
                    "Bojan",
                    query="ok sam sa tech dogadjajima ali volim i kad ima neformalni deo posle, neka zurka ili druzenje",
                    liked_types=["tech", "networking", "social", "bar", "startup"],
                    noise="medium",
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # G09 — Romantična večera / date night                                #
        # ------------------------------------------------------------------ #
        {
            "group_id": "G09_date_night",
            "description": "Par — romantičan izlazak, ambijent je ključan",
            "members": [
                _member(
                    "Nikola",
                    query="planiramo romanticno vece sa devojkom, treba mi nesto posebno, dobar restoran ili event",
                    liked_types=["restaurant", "wine", "romantic", "culture", "live_music"],
                    noise="low",
                    budget_max_rsd=5000,
                ),
                _member(
                    "Jelena",
                    query="volim umetnost i dobru hranu, nesto sa lepim ambijentom, ne mora biti skupo",
                    liked_types=["art", "exhibition", "restaurant", "wine", "acoustic"],
                    noise="low",
                    forbidden=["nightclub", "rave", "techno"],
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # G10 — Penzioneri / vikendom                                         #
        # ------------------------------------------------------------------ #
        {
            "group_id": "G10_penzioneri",
            "description": "Starija generacija — kultura, priroda, miran provod",
            "members": [
                _member(
                    "Slavica",
                    query="trazim nesto kulturno za vikend, izlozba ili pozoriste, mirno i pristupacno",
                    liked_types=["theatre", "exhibition", "museum", "culture"],
                    noise="low",
                    must_be_free=True,
                    forbidden=["nightclub", "rave", "techno", "concert"],
                ),
                _member(
                    "Miodrag",
                    query="volim prirodu i setnju, park ili izlet blizu Beograda, nesto opusteno",
                    liked_types=["nature", "park", "outdoor", "excursion", "hiking"],
                    noise="low",
                    must_be_free=True,
                ),
                _member(
                    "Vesna",
                    query="draga mi je klasicna muzika ili narodni program, nesto sta znam i volim",
                    liked_types=["classical_music", "folk", "concert", "theatre"],
                    noise="low",
                    forbidden=["nightclub", "rave", "techno"],
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # G11 — Sportisti                                                     #
        # ------------------------------------------------------------------ #
        {
            "group_id": "G11_sportisti",
            "description": "Sportski tip — aktivno, takmičarsko, energično",
            "members": [
                _member(
                    "Miloš",
                    query="trazim sportski dogadjaj, trka, turnir ili neka grupna fizicka aktivnost",
                    liked_types=["sports", "fitness", "running", "tournament", "outdoor"],
                    noise="medium_high",
                ),
                _member(
                    "Tamara",
                    query="volim jogu i pilates, ali i trcanje u prirodi, nesto aktivno ali opustajuce",
                    liked_types=["yoga", "fitness", "outdoor", "running", "wellness"],
                    noise="low",
                    forbidden=["nightclub", "rave", "techno"],
                ),
                _member(
                    "Dejan",
                    query="sve mi je ok ako je fizicki aktivno, i team sport i individualni, bitno da se krece",
                    liked_types=["sports", "outdoor", "fitness", "cycling", "hiking"],
                    noise="medium",
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # G12 — Expati u Beogradu                                             #
        # ------------------------------------------------------------------ #
        {
            "group_id": "G12_expati",
            "description": "Stranci u Beogradu — upoznaju grad, traže autentično iskustvo",
            "members": [
                _member(
                    "Emma (Nemačka)",
                    query="I want to experience authentic Belgrade culture, local music or art, something unique",
                    liked_types=["culture", "local", "music", "art", "exhibition"],
                    noise="medium",
                ),
                _member(
                    "James (UK)",
                    query="looking for craft beer spots or local food events, nothing too touristy",
                    liked_types=["beer", "food", "local", "bar", "gastronomy"],
                    noise="medium_high",
                ),
                _member(
                    "Yuki (Japan)",
                    query="interested in galleries, design events, or photography exhibitions in Belgrade",
                    liked_types=["gallery", "design", "exhibition", "photography", "art"],
                    noise="low",
                    forbidden=["nightclub", "rave"],
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # G13 — Kolege posle posla                                            #
        # ------------------------------------------------------------------ #
        {
            "group_id": "G13_kolege_posle_posla",
            "description": "Tim izlazak posle posla — opuštanje, nije previše glasno",
            "members": [
                _member(
                    "Ivana (menadzer)",
                    query="organizujem team building posle posla, nesto opustajuce i drustveno, ne previse glasno",
                    liked_types=["social", "bar", "wine", "lounge", "restaurant"],
                    noise="medium",
                    earliest_hour=17,
                ),
                _member(
                    "Petar (developer)",
                    query="ok sam sa barovima i kaficima, samo da nije techno, hocu da mogu da pricam",
                    liked_types=["bar", "cafe", "social", "quiz", "live_music"],
                    noise="medium",
                    forbidden=["techno", "rave", "nightclub"],
                    earliest_hour=17,
                ),
                _member(
                    "Milena (dizajner)",
                    query="nesto sa lepim ambijentom i dizajnom, kafic sa karakterom ili neka izlozba pa posle pice",
                    liked_types=["cafe", "design", "exhibition", "bar", "aesthetic"],
                    noise="medium",
                    earliest_hour=17,
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # G14 — Mikro-festival vikend                                         #
        # ------------------------------------------------------------------ #
        {
            "group_id": "G14_festival_vikend",
            "description": "Festival vikend — muzika, hrana, umetnost zajedno",
            "members": [
                _member(
                    "Predrag",
                    query="hocu na neki festival ovaj vikend, muzika i hrana, napolju ako moze",
                    liked_types=["festival", "outdoor", "live_music", "food", "open_air"],
                    noise="high",
                ),
                _member(
                    "Nevena",
                    query="festival je super, ali da ima i nesto za kupiti, market ili umetnost uz muziku",
                    liked_types=["festival", "market", "art", "live_music", "outdoor"],
                    noise="medium_high",
                ),
                _member(
                    "Uros",
                    query="volim filmske festivale ili kratke filmove, ali i muzicki festival je ok",
                    liked_types=["film", "festival", "cinema", "live_music", "art"],
                    noise="medium",
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # G15 — Potpuno razliciti ukusi (maksimalni stres test kompromisa)    #
        # ------------------------------------------------------------------ #
        {
            "group_id": "G15_max_kompromis",
            "description": "Maksimalni kompromis — techno fan + tihi introvert + dete + veganka",
            "members": [
                _member(
                    "Darko",
                    query="hocu techno zurku, sto glasniju sto bolju, deep house ili techno klub",
                    liked_types=["techno", "electronic", "club", "rave", "nightlife"],
                    noise="high",
                    earliest_hour=22,
                ),
                _member(
                    "Mirjana",
                    query="ne podnosim buku uopste, hocu nesto tiho, mozda citaonica, joga ili tiha izlozba",
                    liked_types=["yoga", "exhibition", "workshop", "mindfulness", "reading"],
                    noise="low",
                    forbidden=["nightclub", "rave", "techno", "concert", "festival"],
                ),
                _member(
                    "Djordje (sa detetom od 6 god)",
                    query="imam dete, mora biti porodicno i bezbedno, nista sa alkoholom ili bukom",
                    liked_types=["family", "outdoor", "park", "workshop", "nature"],
                    noise="low",
                    forbidden=["nightclub", "rave", "techno", "bar", "club"],
                    must_be_free=True,
                ),
                _member(
                    "Ivana (veganka)",
                    query="bitno mi je da ima vegan hrane, food festival ili restoran sa vegan opcijama",
                    liked_types=["food", "vegan", "food_festival", "outdoor", "market"],
                    noise="medium",
                    forbidden=["nightclub", "rave", "techno"],
                ),
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class GroupResult:
    group_id: str
    description: str
    member_names: list[str]
    candidates: list[tuple[NormalizedEvent, float]]
    ranked: list[tuple[NormalizedEvent, float, float, float]]
    latency_s: float
    hard_pass: int
    hard_total: int


def run_group(
    group: dict,
    events: list[NormalizedEvent],
    corpus_matrix: np.ndarray,
    encoder: SentenceEncoder,
) -> GroupResult:
    members = group["members"]
    hards = [m["prefs"].hard_constraints for m in members]
    prefs = [m["prefs"] for m in members]
    query = "\n\n".join(m["query"] for m in members)
    _liked_terms = " ".join(
        t.replace("_", " ")
        for m in members
        for t in (m["prefs"].soft_preferences.liked_types or [])
    )
    if _liked_terms:
        query += "\n\n" + _liked_terms

    t0 = time.perf_counter()

    # Encode only the query — corpus is already encoded and passed in
    query_vec = encoder.encode([query.strip()])[0]
    pool_hits = cosine_top_k(corpus_matrix, query_vec, k=80)

    # Apply group hard filters on the semantic pool
    candidates: list[tuple[NormalizedEvent, float]] = []
    for idx, score in pool_hits:
        event = events[idx]
        if _is_non_event(event.event_description_resolved, event.tags):
            continue
        attrs = extract_event_attributes_maybe_lazy_gemini(event)
        if passes_all_hard_constraints(hards, event, attrs):
            candidates.append((event, score))
        if len(candidates) >= 20:
            break

    ranked = rank_by_least_misery(candidates, prefs)[:10]
    latency = time.perf_counter() - t0

    hard_pass = sum(
        1 for event, *_ in ranked
        if passes_all_hard_constraints(
            hards, event, extract_event_attributes_maybe_lazy_gemini(event)
        )
    )

    return GroupResult(
        group_id=group["group_id"],
        description=group["description"],
        member_names=[m["name"] for m in members],
        candidates=candidates,
        ranked=ranked,
        latency_s=latency,
        hard_pass=hard_pass,
        hard_total=len(ranked),
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
_SEP = "=" * 72
_SEP_THIN = "-" * 72


def print_report(results: list[GroupResult]) -> None:
    print(f"\n{_SEP}")
    print("  BELGRADE GROUP RECOMMENDER — AUTOMATED TEST REPORT")
    print(_SEP)

    for res in results:
        print(f"\n{_SEP_THIN}")
        print(f"  {res.group_id}  |  {res.description}")
        print(_SEP_THIN)
        print(f"  Members    : {', '.join(res.member_names)}")
        print(f"  Latency    : {res.latency_s:.3f}s")
        print(f"  Candidates : {len(res.candidates)}  (after semantic + hard filter)")
        print(f"  Ranked     : {len(res.ranked)}")

        if res.hard_total > 0:
            rate = 100 * res.hard_pass / res.hard_total
            flag = "OK" if rate == 100 else "WARN"
            print(f"  Hard constraints : {res.hard_pass}/{res.hard_total} ({rate:.0f}%)  [{flag}]")

        if res.ranked:
            ev, sem, mn, mean = res.ranked[0]
            print(f"\n  TOP RESULT:")
            print(f"    sem={sem:.4f}  min_soft={mn:.4f}  mean_soft={mean:.4f}")
            snippet = (ev.event_description_resolved or "")[:120].replace("\n", " ")
            print(f"    {snippet}")
            print(f"    {ev.link}")

            print(f"\n  FULL RANKING:")
            for i, (ev, sem, mn, mean) in enumerate(res.ranked, 1):
                tags_str = ", ".join(ev.tags[:5]) or "—"
                line = (ev.event_description_resolved or "")[:65].replace("\n", " ")
                print(f"    {i:2}. sem={sem:.3f}  min={mn:.3f}  mean={mean:.3f}  [{tags_str}]")
                print(f"        {line}")
        else:
            print("\n  WARNING: No results returned.")

    print(f"\n{_SEP}")
    print("  SUMMARY")
    print(_SEP)
    print(f"  {'Group':<28} {'Latency':>8}  {'Candidates':>10}  {'Hard%':>6}  {'TopMin':>7}  {'TopMean':>8}")
    print(f"  {'-'*28} {'-'*8}  {'-'*10}  {'-'*6}  {'-'*7}  {'-'*8}")
    for res in results:
        hard_pct = f"{100 * res.hard_pass / res.hard_total:.0f}%" if res.hard_total else "N/A"
        top_min = f"{res.ranked[0][2]:.4f}" if res.ranked else "N/A"
        top_mean = f"{res.ranked[0][3]:.4f}" if res.ranked else "N/A"
        print(
            f"  {res.group_id:<28} {res.latency_s:>7.3f}s"
            f"  {len(res.candidates):>10}  {hard_pct:>6}  {top_min:>7}  {top_mean:>8}"
        )
    print(_SEP)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_ROOT / "fixtures" / "events_dataset.jsonl",
    )
    parser.add_argument(
        "--normalized-out",
        type=Path,
        default=_ROOT / "data" / "processed" / "events_normalized_test.jsonl",
    )
    parser.add_argument("--skip-ingest", action="store_true")
    args = parser.parse_args()

    # Step 1: Ingest
    if not args.skip_ingest:
        if not args.dataset.is_file():
            log.error("Dataset not found: %s", args.dataset)
            sys.exit(1)
        log.info("Step 1/3 — Ingesting %s …", args.dataset)
        t0 = time.perf_counter()
        written, skipped = ingest_jsonl_file(args.dataset, args.normalized_out)
        log.info("Done in %.2fs — %d written, %d skipped", time.perf_counter() - t0, written, skipped)
    else:
        if not args.normalized_out.is_file():
            log.error("Normalized file not found: %s", args.normalized_out)
            sys.exit(1)
        log.info("Step 1/3 — Skipping ingest, using %s", args.normalized_out)

    # Step 2: Load corpus
    log.info("Step 2/3 — Loading corpus …")
    all_events = load_normalized_events(args.normalized_out)
    indexable = [e for e in all_events if e.include_in_index]
    log.info("%d total, %d indexable", len(all_events), len(indexable))
    if not indexable:
        log.error("No indexable events.")
        sys.exit(1)

    # Step 3: Load encoder + encode corpus ONCE — all groups share it
    log.info("Step 3/4 — Loading encoder and encoding corpus (one-time) …")
    encoder = SentenceEncoder()
    texts = [build_embedding_text(e) for e in indexable]
    t_enc = time.perf_counter()
    corpus_matrix = encoder.encode(texts)
    log.info(
        "Corpus encoded in %.1fs — shape %s model=%s",
        time.perf_counter() - t_enc, corpus_matrix.shape, encoder.model_name,
    )

    # Step 4: Run groups — only query encoding per group (~instant)
    groups = build_test_groups()
    log.info("Step 4/4 — Running %d test groups …", len(groups))
    results: list[GroupResult] = []
    for group in groups:
        log.info("  %s …", group["group_id"])
        try:
            res = run_group(group, indexable, corpus_matrix, encoder)
            results.append(res)
            log.info("  Done — %d candidates, %d ranked, %.3fs", len(res.candidates), len(res.ranked), res.latency_s)
        except Exception:
            log.exception("  %s FAILED", group["group_id"])

    print_report(results)


if __name__ == "__main__":
    main()
