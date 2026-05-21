#!/usr/bin/env python3
"""
Realistic group scenarios — everyday situations real groups of people in Belgrade face.

No impossible constraints, no wild edge cases. Just real life:
birthday dinners, tourist groups, work lunches, reunions, new neighbors,
musicians, parents night out, first dates, language exchanges.

Usage:
    python run_realistic_tests.py --skip-ingest
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

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
log = logging.getLogger("realtest")

import numpy as np

from belgrade_recommender.extract.attributes import extract_event_attributes_maybe_lazy_gemini
from belgrade_recommender.ingest.models import NormalizedEvent
from belgrade_recommender.ingest.pipeline import ingest_jsonl_file
from belgrade_recommender.rank.group_rank import rank_by_least_misery
from belgrade_recommender.retrieve.cosine_search import cosine_top_k
from belgrade_recommender.retrieve.encoder import SentenceEncoder
from belgrade_recommender.retrieve.hard_filters import passes_all_hard_constraints
from belgrade_recommender.retrieve.service import load_normalized_events
from belgrade_recommender.retrieve.text import build_embedding_text
from belgrade_recommender.schemas.parsed_preferences import (
    ParsedHardConstraints,
    ParsedSoftPreferences,
    ParsedUserPreferences,
)


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


def build_test_groups() -> list[dict]:
    return [

        # ------------------------------------------------------------------ #
        # L01 — Birthday surprise organization                                #
        # Friends planning a surprise birthday night for their friend Jelena  #
        # Organizers have fun + must match Jelena's known preferences         #
        # ------------------------------------------------------------------ #
        {
            "group_id": "L01_rodendan_iznenadjenje",
            "description": "Organizujemo iznenadjenje za Jelenin rodjendan — ona voli jazz i fine dining",
            "members": [
                _member(
                    "Ana (organizator)",
                    query="organizujem iznenadjenje za rodjendan za drugaricu Jelenu, ona obozava jazz muziku i dobru hranu, hocu nesto posebno, ne mora biti skupo ali mora biti memorabilno",
                    liked_types=["jazz", "restaurant", "live_music", "celebration", "fine_dining"],
                    noise="medium",
                    budget_max_rsd=4000,
                    earliest_hour=19,
                    forbidden=["techno", "rave", "nightclub"],
                ),
                _member(
                    "Milica (drugarica)",
                    query="idemo da slavimo rodjendan, volim jazz i dobru atmosferu, nesto gde mozemo da sednemo i uzivamo u vecerasnji program",
                    liked_types=["jazz", "live_music", "restaurant", "wine", "celebration"],
                    noise="medium",
                    budget_max_rsd=4000,
                    earliest_hour=19,
                ),
                _member(
                    "Luka (drugar)",
                    query="slavljenicki vecer, ok sam sa jazz-om, bitno da je dobra atmosfera i da moze da se pije, ne mora biti skupo",
                    liked_types=["bar", "live_music", "jazz", "wine", "social"],
                    noise="medium",
                    budget_max_rsd=3500,
                    earliest_hour=19,
                    forbidden=["techno", "rave"],
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # L02 — Blind date, safe neutral ground                               #
        # Two people meeting for the first time, need a relaxed, not-too-    #
        # loud place where conversation is easy                               #
        # ------------------------------------------------------------------ #
        {
            "group_id": "L02_blind_date",
            "description": "Blind date — prvi put se srecu, treba neutralno mesto, lako za razgovor",
            "members": [
                _member(
                    "Marko (on)",
                    query="imam blind date, zelim nesto opusteno gde mozemo da razgovaramo, dobar kafic ili restoran, nije previse glasno, mozda malo muzike u pozadini",
                    liked_types=["cafe", "restaurant", "wine", "acoustic", "bar"],
                    noise="low",
                    budget_max_rsd=3000,
                    earliest_hour=18,
                    forbidden=["nightclub", "rave", "techno", "karaoke"],
                ),
                _member(
                    "Sara (ona)",
                    query="prvi sastanak sa nekim novim, zelim neutralno mesto gde je prijatno, moze kafic sa karakterom ili kulturno mesto, ne previse formalno ali ni zurka",
                    liked_types=["cafe", "wine_bar", "art", "acoustic", "restaurant"],
                    noise="low",
                    budget_max_rsd=3000,
                    earliest_hour=18,
                    forbidden=["nightclub", "rave", "techno"],
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # L03 — Girls weekend trip to Belgrade                                #
        # Three friends from different cities visiting Belgrade for 2 days,  #
        # want to see the best of the city — mix of culture + fun            #
        # ------------------------------------------------------------------ #
        {
            "group_id": "L03_devojke_vikend_trip",
            "description": "Tri drugarice dolaze u Beograd na vikend — kultura danju, provod navecer",
            "members": [
                _member(
                    "Jovana (iz Novog Sada)",
                    query="dolazim u Beograd na vikend sa drugaricama, zelim da vidim nesto sto nisam videla, muzeji danju i dobar bar ili koktel bar navecer",
                    liked_types=["museum", "exhibition", "culture", "cocktail", "bar", "city_tour"],
                    noise="medium",
                    budget_max_rsd=3000,
                ),
                _member(
                    "Tamara (iz Nis)",
                    query="vikend u Beogradu, hocu autenticno belgrade iskustvo, lokalni barovi, dobra hrana, malo kulture malo provoda",
                    liked_types=["local", "bar", "food", "culture", "authentic", "nightlife"],
                    noise="medium_high",
                    budget_max_rsd=3500,
                ),
                _member(
                    "Ivana (iz Kragujevca)",
                    query="prvi put u Beogradu na ovako kratkom tripu, zelim da vidim sto vise, nesto vizualno interesantno za Instagram i dobar provod",
                    liked_types=["aesthetic", "art", "bar", "nightlife", "food", "design"],
                    noise="medium_high",
                    budget_max_rsd=3000,
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # L04 — Students celebrating end of exam period                       #
        # Budget very tight, want to let loose after a stressful month       #
        # ------------------------------------------------------------------ #
        {
            "group_id": "L04_studenti_kraj_ispita",
            "description": "Studenti slavе kraj ispitnog roka — budzet mali, energija visoka",
            "members": [
                _member(
                    "Petar (student fizike)",
                    query="zavrsio sam sve ispite! slavimo sa drugarima, budzet je mali ali hocu dobru zurku ili koncert, nesto sto ce biti zapamceno",
                    liked_types=["party", "concert", "bar", "festival", "live_music"],
                    noise="high",
                    budget_max_rsd=1500,
                    earliest_hour=20,
                    must_be_free=False,
                ),
                _member(
                    "Milena (studentkinja dizajna)",
                    query="kraj ispita yay! hocu nesto vizualno interesantno, maybe neka izlozba pa posle bar, budzet je ogranicen",
                    liked_types=["exhibition", "art", "bar", "concert", "design"],
                    noise="medium_high",
                    budget_max_rsd=1500,
                    earliest_hour=18,
                ),
                _member(
                    "Nikola (student ekonomije)",
                    query="slavimo kraj ispita, budzetno ali zabavno, zurka ili bar ili koncert, nema mnogo para ali hocu da se provedem",
                    liked_types=["bar", "party", "concert", "festival", "social"],
                    noise="high",
                    budget_max_rsd=1500,
                    earliest_hour=20,
                ),
                _member(
                    "Анна (студентка из России)",
                    query="конец сессии! хочу отметить с друзьями, бюджет небольшой, хочу куда-то с хорошей атмосферой и музыкой",
                    liked_types=["party", "bar", "concert", "live_music", "social"],
                    noise="high",
                    budget_max_rsd=1500,
                    earliest_hour=20,
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # L05 — International company team outing                             #
        # Diverse team, after-work drinks + dinner, everyone must be comfy   #
        # ------------------------------------------------------------------ #
        {
            "group_id": "L05_tim_medjunarodna_firma",
            "description": "Medjunarodni tim ide na izlazak posle posla — vise kultura, jedna vecer",
            "members": [
                _member(
                    "Milica (HR, Srbija)",
                    query="organizujem team izlazak za medjunarodnu ekipu, razlicite kulture i ukusi, hocu nesto gde ce se svi osecati prijatno, dobra hrana, nije previse glasno",
                    liked_types=["restaurant", "wine", "social", "food", "bar"],
                    noise="medium",
                    earliest_hour=18,
                    forbidden=["nightclub", "rave", "techno"],
                    budget_max_rsd=4000,
                ),
                _member(
                    "James (UK, developer)",
                    query="team outing, I want good local Serbian food and craft beer, relaxed atmosphere, not too noisy",
                    liked_types=["food", "craft_beer", "bar", "local", "restaurant"],
                    noise="medium",
                    earliest_hour=18,
                    budget_max_rsd=4000,
                ),
                _member(
                    "Fatima (marketing, Maroko)",
                    query="team building izlazak, trazim nesto gde moze i bezalkoholno, dobra hrana, kultura je plus",
                    liked_types=["restaurant", "food", "culture", "non_alcoholic", "social"],
                    noise="medium",
                    earliest_hour=18,
                    budget_max_rsd=3500,
                    forbidden=["nightclub", "rave"],
                ),
                _member(
                    "Дмитрий (backend, Русия)",
                    query="корпоратив после работы, хочу хорошего вина или пива, местная еда, не слишком шумно, чтобы можно было поговорить",
                    liked_types=["wine", "food", "restaurant", "bar", "social"],
                    noise="medium",
                    earliest_hour=18,
                    forbidden=["nightclub", "rave", "techno"],
                    budget_max_rsd=4000,
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # L06 — Parents night out (young parents, rare free evening)          #
        # They have a babysitter until midnight — want to make it count       #
        # ------------------------------------------------------------------ #
        {
            "group_id": "L06_mladi_roditelji_noc",
            "description": "Mladi roditelji imaju bebisiterku do ponoci — retka slobodna vecer, hoce da uzivaju",
            "members": [
                _member(
                    "Ivana (mama)",
                    query="imamo bebisiterku do ponoci, retko izlazimo, hocu nesto posebno sa muzem, dobar restoran ili koncert, nesto za odrasle, ne mora biti skupo ali da je posebno",
                    liked_types=["restaurant", "live_music", "wine", "concert", "jazz"],
                    noise="medium",
                    budget_max_rsd=5000,
                    earliest_hour=19,
                    forbidden=["techno", "rave", "nightclub"],
                ),
                _member(
                    "Uroš (tata)",
                    query="slobodna vecer bez dece, hocu nesto opusteno sa zenom, dobar obrok ili ziva muzika, nije me briga za zanr samo da je lepa atmosfera",
                    liked_types=["restaurant", "bar", "live_music", "wine", "acoustic"],
                    noise="medium",
                    budget_max_rsd=5000,
                    earliest_hour=19,
                    forbidden=["rave", "nightclub"],
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # L07 — Old friends reunion after 3 years abroad                      #
        # One friend just came back from living abroad, wants to rediscover  #
        # Belgrade together with the group                                    #
        # ------------------------------------------------------------------ #
        {
            "group_id": "L07_reunjion_iz_inostranstva",
            "description": "Drug se vratio iz Berlina posle 3 godine — pokazujemo mu novi Beograd",
            "members": [
                _member(
                    "Stefan (vratio se iz Berlina)",
                    query="vratio sam se iz Berlina posle 3 godine, hocu da vidim sta je novo u Beogradu, nova mesta, nova kultura, sta smo propustio, iznenadite me",
                    liked_types=["new_venue", "culture", "bar", "music", "art", "local"],
                    noise="medium_high",
                    budget_max_rsd=3000,
                    earliest_hour=19,
                ),
                _member(
                    "Nikola (domacin, zivi u BG)",
                    query="pokazujem Stefanu novi Beograd, hocu neka nova mesta koja su se otvorila, kreativna scena, nesto sto nije postojalo pre 3 godine",
                    liked_types=["new_venue", "art", "bar", "culture", "creative", "music"],
                    noise="medium_high",
                    budget_max_rsd=3000,
                ),
                _member(
                    "Maja (drugarica)",
                    query="drago mi je sto se Stefan vratio, hocu neku dobru vecer sa drustvom, muzika i pice, noviji lokali u Beogradu",
                    liked_types=["bar", "music", "social", "live_music", "new_venue"],
                    noise="medium_high",
                    budget_max_rsd=3000,
                    earliest_hour=20,
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # L08 — First day in Belgrade (tourists, just arrived)                #
        # Classic tourist group — want the highlights, orientation, local    #
        # food, and something authentic for their first evening              #
        # ------------------------------------------------------------------ #
        {
            "group_id": "L08_turisti_prvi_dan",
            "description": "Turisti koji su tek stigli — prvi dan u Beogradu, hoce autenticno iskustvo",
            "members": [
                _member(
                    "Laura (Spanija)",
                    query="just arrived in Belgrade, first evening, we want authentic local food and drinks, something that shows us the real Belgrade, not touristy",
                    liked_types=["local", "food", "restaurant", "authentic", "culture", "serbian"],
                    noise="medium",
                    budget_max_rsd=3000,
                    earliest_hour=18,
                ),
                _member(
                    "Kenji (Japan)",
                    query="first night in Belgrade, interested in local culture and food, maybe a walk and then dinner somewhere historic or authentic",
                    liked_types=["local", "culture", "food", "historic", "outdoor", "restaurant"],
                    noise="low",
                    budget_max_rsd=3000,
                    earliest_hour=18,
                ),
                _member(
                    "Katia (Rusija)",
                    query="первый день в Белграде, хочу попробовать настоящую сербскую кухню и увидеть что-то интересное вечером, не туристические ловушки",
                    liked_types=["local", "food", "restaurant", "culture", "authentic", "serbian"],
                    noise="medium",
                    budget_max_rsd=3000,
                    earliest_hour=18,
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # L09 — Book club monthly dinner                                      #
        # Literary group that meets once a month, current book is about      #
        # Balkans, want a themed dinner with good conversation atmosphere    #
        # ------------------------------------------------------------------ #
        {
            "group_id": "L09_knjizevni_klub_vecera",
            "description": "Knjizevni klub mesecna vecera — citali Balkanska proza, hoce autenticni ambijent za razgovor",
            "members": [
                _member(
                    "Gordana (profesorka knjizevnosti)",
                    query="vodim knjizevni klub, ovaj mesec smo citali balkanske autore, trazim restoran ili kafanu sa dobrim ambijentom gde mozemo da razgovaramo tiho i udobno, srpska hrana ili balkanska kuhinja",
                    liked_types=["restaurant", "kafana", "traditional", "food", "quiet", "culture"],
                    noise="low",
                    budget_max_rsd=2500,
                    earliest_hour=18,
                    forbidden=["nightclub", "rave", "concert"],
                ),
                _member(
                    "Biljana (clanica kluba)",
                    query="mesecni sastanak knjizevnog kluba, volim mesta sa karakterom i istorijom, kafana ili restoran gde moze da se sedi dugo i razgovara",
                    liked_types=["kafana", "restaurant", "traditional", "culture", "wine"],
                    noise="low",
                    budget_max_rsd=2500,
                    earliest_hour=18,
                    forbidden=["nightclub", "rave"],
                ),
                _member(
                    "Markus (Nemacka, clanica kluba)",
                    query="book club dinner, looking for a traditional Serbian restaurant or tavern, authentic, good for long conversations, Balkan atmosphere",
                    liked_types=["restaurant", "traditional", "serbian", "kafana", "authentic"],
                    noise="low",
                    budget_max_rsd=2500,
                    earliest_hour=18,
                    forbidden=["nightclub", "rave", "nightlife"],
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # L10 — Running club after the race                                   #
        # Group of runners after a Sunday morning race, want brunch/lunch    #
        # somewhere they can come in sweaty running gear, casual, hungry     #
        # ------------------------------------------------------------------ #
        {
            "group_id": "L10_trkacu_posle_trke",
            "description": "Trkaci posle nedeljne trke — gladni, u trenirci, traze brunch ili rucak",
            "members": [
                _member(
                    "Jelena (organizatorka trke)",
                    query="zavrsili smo trcanje, glad nas je savladala, hocu mesto gde mogu da dodjemo u sportskoj opremi bez da nas gledaju cudno, dobar brunch ili rucak, nesto zdravo ali i sitno",
                    liked_types=["brunch", "healthy_food", "cafe", "restaurant", "casual", "breakfast"],
                    noise="medium",
                    budget_max_rsd=1500,
                    forbidden=["nightclub", "formal", "fine_dining"],
                ),
                _member(
                    "Miloš (maraton runner)",
                    query="posle trke zelim proteinski obrok, zdravo ali i ukusno, casual kafic ili restoran, nismo stavljeni za formalno",
                    liked_types=["healthy_food", "brunch", "cafe", "sports_nutrition", "casual"],
                    noise="medium",
                    budget_max_rsd=1500,
                    forbidden=["nightclub", "formal"],
                ),
                _member(
                    "Olga (tri runner)",
                    query="после пробежки хочу хорошего завтрака или бранча, здоровая еда, кафе где можно прийти в спортивной одежде",
                    liked_types=["brunch", "healthy_food", "cafe", "breakfast", "casual"],
                    noise="medium",
                    budget_max_rsd=1500,
                    forbidden=["nightclub", "formal", "fine_dining"],
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # L11 — Art students looking for inspiration                          #
        # Fine arts / design students seeking new visual input, free if      #
        # possible, they want to sketch and observe, not just party          #
        # ------------------------------------------------------------------ #
        {
            "group_id": "L11_umetnici_studenti",
            "description": "Studenti umetnosti traze inspiraciju — izlozbe, ulicna umetnost, ateljeи",
            "members": [
                _member(
                    "Bojana (slikarstvo)",
                    query="studentkinja sam likovne umetnosti, trazim inspiraciju, izlozbe, otvorene atelije ili street art u Beogradu, besplatno je plus, hocu nesto sto cu da zapamtim",
                    liked_types=["exhibition", "art", "gallery", "street_art", "atelier", "design"],
                    noise="low",
                    must_be_free=True,
                    forbidden=["nightclub", "rave"],
                ),
                _member(
                    "Radan (graficki dizajn)",
                    query="dizajner sam, hocu da vidim dobru tipografiju, vizuelni dizajn, savremenu umetnost, sto vizuelno stimulativnije utoliko bolje, besplatno preferred",
                    liked_types=["design", "typography", "exhibition", "art", "gallery", "contemporary_art"],
                    noise="low",
                    must_be_free=True,
                    forbidden=["nightclub", "rave"],
                ),
                _member(
                    "Hana (fotografija)",
                    query="fotografkinja, trazim izlozbe fotografije, otvorene studije, ili mesta koja fotografima izgledaju interesantno, urban landscape, arhitektura",
                    liked_types=["photography", "exhibition", "art", "urban", "architecture", "gallery"],
                    noise="low",
                    must_be_free=True,
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # L12 — Local showing Belgrade to a foreign friend                    #
        # Classic scenario: local expert + curious foreigner, full day plan  #
        # ------------------------------------------------------------------ #
        {
            "group_id": "L12_domacin_i_gost",
            "description": "Srbin pokazuje Beograd stranom prijatelju — jutro do ponoci, sve najlepse",
            "members": [
                _member(
                    "Aleksandar (domacin, BG)",
                    query="vodim stranog prijatelja kroz Beograd, hocu da mu pokazem sve sto volim — dobru kafu, kulturna mesta, stari grad, pa vecer sa muzikom i pice",
                    liked_types=["cafe", "culture", "history", "bar", "live_music", "local", "outdoor"],
                    noise="medium",
                    budget_max_rsd=3000,
                ),
                _member(
                    "Antoine (Francuz, gost)",
                    query="Belgrade tourist, my Serbian friend is showing me the city, I want to understand Serbian culture, food, music — the real thing, not tourist traps",
                    liked_types=["local", "culture", "food", "music", "authentic", "serbian", "history"],
                    noise="medium",
                    budget_max_rsd=3000,
                    forbidden=["tourist", "mainstream"],
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # L13 — Half the table vegan, half not                               #
        # Common real situation: mixed dietary group going out together       #
        # ------------------------------------------------------------------ #
        {
            "group_id": "L13_pola_vegani_pola_ne",
            "description": "Polovina grupe je vegan, polovina jede meso — traze restoran koji nudi oboje",
            "members": [
                _member(
                    "Dragana (veganka)",
                    query="ja sam veganka, trazim restoran u Beogradu gde ima dobra vegan ponuda, ne samo salata, pravo vegan jelo, ali da moze i ne-vegani da jedu sa mnom",
                    liked_types=["vegan", "restaurant", "food", "plant_based", "healthy"],
                    noise="medium",
                    forbidden=["rave", "nightclub"],
                ),
                _member(
                    "Nemanja (mesojed)",
                    query="idem na rucak sa vegan drugaricama, ok mi je da jele vegan ali ja zelim normalnu hranu, restoran koji ima i jedno i drugo",
                    liked_types=["restaurant", "food", "grill", "meat", "local"],
                    noise="medium",
                    budget_max_rsd=2000,
                ),
                _member(
                    "Iva (fleksitarijanac)",
                    query="jedem i meso i vegan hranu, bitno mi je da restoran ima raznovrstan meni i dobru atmosferu, nesto prijatno i ukusno",
                    liked_types=["restaurant", "food", "diverse_menu", "wine", "social"],
                    noise="medium",
                    budget_max_rsd=2000,
                ),
                _member(
                    "Маша (веган)",
                    query="я веган, ищу ресторан в Белграде с хорошим веганским меню, но чтобы не-веганы тоже могли поесть, хочу провести вечер с друзьями",
                    liked_types=["vegan", "restaurant", "plant_based", "food", "social"],
                    noise="medium",
                    forbidden=["nightclub", "rave"],
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # L14 — Quick office lunch, back in an hour                           #
        # Weekday lunch, need something fast, good, close to city center,    #
        # back at desk within 60-70 minutes                                  #
        # ------------------------------------------------------------------ #
        {
            "group_id": "L14_brzi_rucak_iz_kancelarije",
            "description": "Brzi rucak iz kancelarije — za sat vremena nazad za stolom, centar Beograda",
            "members": [
                _member(
                    "Katarina (menadzer)",
                    query="brzi radni rucak sa kolegama, imamo sat vremena, trazim nesto brzo i ukusno u centru Beograda, ne formalno ali ni fast food, dobar rucak za pare",
                    liked_types=["lunch", "restaurant", "fast_casual", "food", "city_center"],
                    noise="medium",
                    budget_max_rsd=1200,
                    forbidden=["nightclub", "formal", "slow_food"],
                ),
                _member(
                    "Boris (developer)",
                    query="rucak za 45 minuta, hocu nesto sito i ukusno, ne hocu da cekam, brzo serviranje, dobar odnos cena-kvalitet",
                    liked_types=["lunch", "food", "fast_casual", "restaurant", "value"],
                    noise="medium",
                    budget_max_rsd=1000,
                ),
                _member(
                    "Emma (dizajner, UK)",
                    query="quick lunch break, want something tasty and not too expensive, fast service, good food, central Belgrade",
                    liked_types=["lunch", "food", "fast_casual", "restaurant", "healthy"],
                    noise="medium",
                    budget_max_rsd=1200,
                    forbidden=["formal", "fine_dining"],
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # L15 — Pre-wedding celebration (mixed gender, groom + friends)       #
        # More relaxed than bachelorette — craft beer, games, memories       #
        # ------------------------------------------------------------------ #
        {
            "group_id": "L15_momacko_vece",
            "description": "Momacko vece — relaxed, pivo, igre, lepe uspomene pre vencanuca",
            "members": [
                _member(
                    "Dušan (mladozenja)",
                    query="momacko vece, hocu nesto opusteno sa drugarima, dobro pivo, malo igara, da se smejemo i da zapamtimo ovu noc, ne mora biti ludo",
                    liked_types=["bar", "craft_beer", "social", "games", "live_music"],
                    noise="medium_high",
                    budget_max_rsd=2000,
                    earliest_hour=19,
                ),
                _member(
                    "Vladan (kum)",
                    query="organizujem momacko, hocu dobar bar ili pub gde mozemo da se smestimo u grupi, dobro pivo i neka zabava",
                    liked_types=["pub", "bar", "craft_beer", "social", "games", "live_music"],
                    noise="medium_high",
                    budget_max_rsd=2000,
                    earliest_hour=19,
                    forbidden=["techno", "rave"],
                ),
                _member(
                    "Jovan (drugar)",
                    query="momacko vece, ok mi je sa svim, glavno da je dobra atmosfera, pivo i dosta smeha, ne hocu previse ozbiljno mesto",
                    liked_types=["bar", "pub", "beer", "social", "fun", "live_music"],
                    noise="medium_high",
                    budget_max_rsd=2000,
                    earliest_hour=19,
                ),
                _member(
                    "Ivan (drugar iz inostranstva)",
                    query="bachelor party, want a fun bar or pub with craft beer, not a club, something where we can talk and have fun",
                    liked_types=["pub", "bar", "craft_beer", "social", "games"],
                    noise="medium_high",
                    budget_max_rsd=2000,
                    earliest_hour=19,
                    forbidden=["techno", "rave", "nightclub"],
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # L16 — Neighbors community building, all ages mixed                  #
        # Organized by a building residents' committee, want something for   #
        # everyone — kids + adults + elderly, Novi Beograd area              #
        # ------------------------------------------------------------------ #
        {
            "group_id": "L16_komsiлuk_event",
            "description": "Susedski izlazak — sve generacije, deca + odrasli + penzioneri, Novi Beograd",
            "members": [
                _member(
                    "Vera (organizator, 60g)",
                    query="organizujem izlazak za komsiluk, ima dece, odraslih i penzionera, trazim nesto na otvorenom gde ima mesta za sve, park ili festival, besplatno bi bilo idealno",
                    liked_types=["outdoor", "park", "festival", "family", "free", "community"],
                    noise="medium",
                    must_be_free=True,
                    forbidden=["nightclub", "rave", "techno"],
                ),
                _member(
                    "Ana (mama malih dece)",
                    query="dolazim sa dvoje male dece, mora biti porodicno i bezbedno, neko zelenilo ili park, deca da se istrce",
                    liked_types=["family", "outdoor", "park", "kids", "nature", "free"],
                    noise="medium",
                    must_be_free=True,
                    forbidden=["nightclub", "rave", "techno", "bar"],
                ),
                _member(
                    "Miloš (tinejdzer)",
                    query="moram da idem sa komsilukom, ok ako ima neke aktivnosti, muzika ili nesto interesantno, ne muzej",
                    liked_types=["outdoor", "festival", "music", "sports", "social"],
                    noise="medium_high",
                    must_be_free=True,
                    forbidden=["museum", "opera", "classical"],
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # L17 — Language exchange meetup                                      #
        # People wanting to practice different languages while socializing:  #
        # Serbian, English, Russian speakers together                        #
        # ------------------------------------------------------------------ #
        {
            "group_id": "L17_razmena_jezika",
            "description": "Razmena jezika — Srbi uce engleski i ruski, stranci uce srpski, druzenje",
            "members": [
                _member(
                    "Milica (uci engleski)",
                    query="trazim language exchange events u Beogradu gde mogu da vezbam engleski sa native speakerima, kafic ili bar gde se susrecu ljudi razlicitih nacija",
                    liked_types=["social", "cafe", "bar", "international", "language", "community"],
                    noise="medium",
                    budget_max_rsd=1500,
                    forbidden=["nightclub", "rave"],
                ),
                _member(
                    "Jake (learning Serbian)",
                    query="I want to practice my Serbian while meeting locals, looking for a language exchange event or social meetup in a cafe or bar",
                    liked_types=["social", "cafe", "bar", "community", "language", "expat"],
                    noise="medium",
                    budget_max_rsd=1500,
                    forbidden=["nightclub", "rave"],
                ),
                _member(
                    "Ксения (учит сербский)",
                    query="ищу языковой обмен или социальные мероприятия где русские и сербы общаются, хочу практиковать сербский",
                    liked_types=["social", "cafe", "bar", "community", "language", "international"],
                    noise="medium",
                    budget_max_rsd=1500,
                    forbidden=["nightclub", "rave"],
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # L18 — New in Belgrade, want to meet locals                          #
        # People who recently moved to Belgrade and want to build a social   #
        # life — expats and newly arrived Serbians                           #
        # ------------------------------------------------------------------ #
        {
            "group_id": "L18_novi_u_gradu",
            "description": "Novi u Beogradu — traze dogadjaje gde mogu da upoznaju ljude i izgrades mreze",
            "members": [
                _member(
                    "Priya (India, 2 months in BG)",
                    query="I moved to Belgrade 2 months ago and don't know many people, looking for social events, expat meetups or community gatherings where I can make friends",
                    liked_types=["social", "networking", "community", "expat", "bar", "meetup"],
                    noise="medium",
                    budget_max_rsd=2000,
                    forbidden=["nightclub", "rave"],
                ),
                _member(
                    "Lazar (dosao iz Nis u BG)",
                    query="nedavno sam se preselio u Beograd iz Nisa, hocu da upoznam ljude, drustveni dogadjaji, kafici gde se srece nova ekipa, neki events za upoznavanje",
                    liked_types=["social", "community", "bar", "meetup", "networking"],
                    noise="medium",
                    budget_max_rsd=1500,
                    forbidden=["nightclub", "rave"],
                ),
                _member(
                    "Марина (переехала из Москвы)",
                    query="переехала в Белград из Москвы, хочу познакомиться с людьми, ищу общественные мероприятия, бары для знакомств, экспат-тусовки",
                    liked_types=["social", "expat", "community", "bar", "meetup", "networking"],
                    noise="medium",
                    budget_max_rsd=2000,
                    forbidden=["nightclub", "rave"],
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # L19 — Open mic and jam session seekers                              #
        # Musicians (amateur and semi-pro) wanting to play or watch open mic #
        # ------------------------------------------------------------------ #
        {
            "group_id": "L19_muzicari_open_mic",
            "description": "Muzicari traze open mic ili jam session u Beogradu — hoce da sviraju ili gledaju",
            "members": [
                _member(
                    "Nemanja (gitarista)",
                    query="gitarista sam, trazim open mic ili jam session u Beogradu gde mogu da sviram, nije bitno koji zanr, hocu zivу interakciju sa publikom",
                    liked_types=["open_mic", "live_music", "jam_session", "bar", "acoustic", "concert"],
                    noise="medium",
                    earliest_hour=19,
                    forbidden=["techno", "rave", "nightclub"],
                ),
                _member(
                    "Lena (pevaçica)",
                    query="pevacica sam, trazim open mic event gde mogu da nastu pim, ili bar da gledam druge amatere, dobra atmosfera za zivi nastup",
                    liked_types=["open_mic", "live_music", "acoustic", "bar", "singer_songwriter"],
                    noise="medium",
                    earliest_hour=19,
                    forbidden=["techno", "rave"],
                ),
                _member(
                    "Tom (drummer, UK)",
                    query="looking for a jam session or open mic night in Belgrade where musicians get together, I play drums and want to meet other musicians",
                    liked_types=["jam_session", "open_mic", "live_music", "bar", "music"],
                    noise="medium_high",
                    earliest_hour=19,
                    forbidden=["techno", "rave"],
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # L20 — Sunday family brunch, three generations                       #
        # Grandparents + parents + kids, Sunday tradition, warm and easy     #
        # ------------------------------------------------------------------ #
        {
            "group_id": "L20_porodicni_nedeljni_brunch",
            "description": "Porodicni nedeljni brunch — baka i deka + roditelji + deca, toplo i opusteno",
            "members": [
                _member(
                    "Snezana (baka, 68g)",
                    query="idemo na nedeljni rucak sa celom porodicom, uključujuci unučad, zelim nesto toplo i prijatno, srpska hrana, da moze svako da nadje nesto za sebe, nije glasno",
                    liked_types=["restaurant", "traditional", "serbian", "family", "food", "lunch"],
                    noise="low",
                    budget_max_rsd=2000,
                    forbidden=["nightclub", "rave", "techno", "concert"],
                ),
                _member(
                    "Bojana (mama, 38g)",
                    query="nedeljni porodicni rucak, dolaze i roditelji i deca, trazim restoran gde ce svima biti prijatno, od bake do unucadi, domaca srpska kuhinja ili balkanska, toplo i opusteno",
                    liked_types=["restaurant", "family", "traditional", "food", "serbian", "outdoor"],
                    noise="medium",
                    budget_max_rsd=2500,
                    forbidden=["nightclub", "rave"],
                ),
                _member(
                    "Nikola (tata, 40g)",
                    query="porodicni rucak u nedelu, mora biti dete-friendly i da ima srpsku hranu, nesto gde mozemo da sednemo dugo bez pozurivanja",
                    liked_types=["restaurant", "family", "traditional", "food", "relaxed"],
                    noise="medium",
                    budget_max_rsd=2500,
                    forbidden=["nightclub", "rave", "techno"],
                ),
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Runner + Report (same structure as previous scripts)
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
    query_vec = encoder.encode([query.strip()])[0]
    pool_hits = cosine_top_k(corpus_matrix, query_vec, k=80)

    candidates: list[tuple[NormalizedEvent, float]] = []
    for idx, score in pool_hits:
        event = events[idx]
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


_SEP = "=" * 72
_SEP_THIN = "-" * 72


def print_report(results: list[GroupResult]) -> None:
    print(f"\n{_SEP}")
    print("  BELGRADE GROUP RECOMMENDER — REALISTIC SCENARIO REPORT")
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
    print(f"  {'Group':<32} {'Lat':>7}  {'Cands':>5}  {'Hard%':>6}  {'TopMin':>7}  {'TopMean':>8}")
    print(f"  {'-'*32} {'-'*7}  {'-'*5}  {'-'*6}  {'-'*7}  {'-'*8}")
    for res in results:
        hard_pct = f"{100 * res.hard_pass / res.hard_total:.0f}%" if res.hard_total else "N/A"
        top_min = f"{res.ranked[0][2]:.4f}" if res.ranked else "N/A"
        top_mean = f"{res.ranked[0][3]:.4f}" if res.ranked else "N/A"
        print(
            f"  {res.group_id:<32} {res.latency_s:>6.3f}s"
            f"  {len(res.candidates):>5}  {hard_pct:>6}  {top_min:>7}  {top_mean:>8}"
        )

    # Quality check
    print(f"\n{_SEP}")
    total = len(results)
    with_results = sum(1 for r in results if r.ranked)
    full_10 = sum(1 for r in results if len(r.ranked) == 10)
    hard_ok = sum(1 for r in results if r.hard_total > 0 and r.hard_pass == r.hard_total)
    avg_lat = sum(r.latency_s for r in results) / total if total else 0
    avg_cands = sum(len(r.candidates) for r in results) / total if total else 0
    print(f"  Groups with results   : {with_results}/{total}")
    print(f"  Groups with full top10: {full_10}/{total}")
    print(f"  Groups 100% hard OK   : {hard_ok}/{total}")
    print(f"  Avg latency per group : {avg_lat:.3f}s")
    print(f"  Avg candidates        : {avg_cands:.1f}")
    print(_SEP)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path,
        default=_ROOT / "fixtures" / "events_dataset.jsonl",
    )
    parser.add_argument(
        "--normalized-out", type=Path,
        default=_ROOT / "data" / "processed" / "events_normalized_test.jsonl",
    )
    parser.add_argument("--skip-ingest", action="store_true")
    args = parser.parse_args()

    if not args.skip_ingest:
        if not args.dataset.is_file():
            log.error("Dataset not found: %s", args.dataset)
            sys.exit(1)
        log.info("Step 1/4 — Ingesting %s …", args.dataset)
        t0 = time.perf_counter()
        written, skipped = ingest_jsonl_file(args.dataset, args.normalized_out)
        log.info("Done in %.2fs — %d written, %d skipped", time.perf_counter() - t0, written, skipped)
    else:
        if not args.normalized_out.is_file():
            log.error("Normalized file not found: %s", args.normalized_out)
            sys.exit(1)
        log.info("Step 1/4 — Skipping ingest, using %s", args.normalized_out)

    log.info("Step 2/4 — Loading corpus …")
    all_events = load_normalized_events(args.normalized_out)
    indexable = [e for e in all_events if e.include_in_index]
    log.info("%d total, %d indexable", len(all_events), len(indexable))
    if not indexable:
        log.error("No indexable events.")
        sys.exit(1)

    log.info("Step 3/4 — Loading encoder and encoding corpus (one-time) …")
    encoder = SentenceEncoder()
    texts = [build_embedding_text(e) for e in indexable]
    t_enc = time.perf_counter()
    corpus_matrix = encoder.encode(texts)
    log.info(
        "Corpus encoded in %.1fs — shape %s  model=%s",
        time.perf_counter() - t_enc, corpus_matrix.shape, encoder.model_name,
    )

    groups = build_test_groups()
    log.info("Step 4/4 — Running %d realistic scenario groups …", len(groups))
    results: list[GroupResult] = []
    for group in groups:
        log.info("  %s …", group["group_id"])
        try:
            res = run_group(group, indexable, corpus_matrix, encoder)
            results.append(res)
            log.info(
                "  Done — %d candidates, %d ranked, %.3fs",
                len(res.candidates), len(res.ranked), res.latency_s,
            )
        except Exception:
            log.exception("  %s FAILED", group["group_id"])

    print_report(results)


if __name__ == "__main__":
    main()
