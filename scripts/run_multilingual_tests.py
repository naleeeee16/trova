#!/usr/bin/env python3
"""
Multilingual automated pipeline test — Russian / Serbian / English group chats.

Covers: language mix, location/vibe queries, music taste, date-specific constraints.

Usage:
    python run_multilingual_tests.py
    python run_multilingual_tests.py --skip-ingest
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Auto-load .env
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
log = logging.getLogger("mltest")

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
from belgrade_recommender.schemas.event_attributes import EventAttributes
from belgrade_recommender.schemas.parsed_preferences import (
    ParsedHardConstraints,
    ParsedSoftPreferences,
    ParsedUserPreferences,
)


# ---------------------------------------------------------------------------
# Helper
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
# Test groups — multilingual, location/vibe, music, date-focused
# ---------------------------------------------------------------------------

def build_test_groups() -> list[dict]:
    return [

        # ------------------------------------------------------------------ #
        # R01 — Russian speakers in Belgrade (all Russian)                    #
        # Three Russian expats/tourists exploring the city                    #
        # ------------------------------------------------------------------ #
        {
            "group_id": "R01_russkie_v_Belgrade",
            "description": "Русские в Белграде — джаз, аутентичный ужин, галерея",
            "members": [
                _member(
                    "Дмитрий",
                    query="ищу джаз-концерт или живую музыку в Белграде сегодня вечером, желательно недорого",
                    liked_types=["jazz", "live_music", "concert", "bar"],
                    noise="medium",
                    earliest_hour=19,
                ),
                _member(
                    "Наташа",
                    query="хочу попробовать аутентичную сербскую кухню, хороший ресторан с национальными блюдами",
                    liked_types=["food", "restaurant", "traditional", "gastronomy"],
                    noise="low",
                    forbidden=["nightclub", "rave", "techno"],
                ),
                _member(
                    "Алексей",
                    query="интересуют выставки и галереи современного искусства, можно и бесплатно",
                    liked_types=["gallery", "exhibition", "art", "museum", "culture"],
                    noise="low",
                    must_be_free=True,
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # R02 — Mixed Russian + Serbian conversation                          #
        # Half the group speaks Serbian, half Russian                         #
        # ------------------------------------------------------------------ #
        {
            "group_id": "R02_srpsko_ruska_ekipa",
            "description": "Mešovita ekipa — Srbi i Rusi zajedno izlaze",
            "members": [
                _member(
                    "Milica",
                    query="hoću nešto živahno večeras, koncerti ili bar sa dobrom muzikom, nije mi važan žanr",
                    liked_types=["concert", "bar", "live_music", "festival"],
                    noise="medium_high",
                    earliest_hour=20,
                ),
                _member(
                    "Катя",
                    query="хочу что-нибудь живое — концерт или вечеринка, желательно не слишком громко",
                    liked_types=["live_music", "concert", "bar", "party"],
                    noise="medium",
                    earliest_hour=19,
                ),
                _member(
                    "Nikola",
                    query="ok sam sa svim, jedino ne volim techno, hoću da mogu da pričam s društvom",
                    liked_types=["bar", "social", "live_music", "lounge"],
                    noise="medium",
                    forbidden=["techno", "rave"],
                ),
                _member(
                    "Иван",
                    query="ищу что-то интересное в Белграде, бар с атмосферой или живая музыка",
                    liked_types=["bar", "live_music", "culture", "social"],
                    noise="medium",
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # R03 — English + Serbian mix, music discovery                        #
        # Expat and local looking for new music                               #
        # ------------------------------------------------------------------ #
        {
            "group_id": "R03_english_srpski_muzika",
            "description": "Expat + local — discovering Belgrade music scene",
            "members": [
                _member(
                    "Sarah (UK)",
                    query="I just moved to Belgrade, looking for indie or alternative rock gigs, something local and underground",
                    liked_types=["indie", "rock", "concert", "live_music", "alternative"],
                    noise="medium_high",
                    earliest_hour=20,
                ),
                _member(
                    "Marko",
                    query="volim indie i alternativni rok, ima dosta domaćih bendova, tražimo živi nastup",
                    liked_types=["rock", "indie", "live_music", "concert", "alternative"],
                    noise="medium_high",
                ),
                _member(
                    "Tom (USA)",
                    query="open to anything with a real live band, no DJ sets please, good sound quality matters",
                    liked_types=["live_music", "concert", "rock", "jazz", "acoustic"],
                    noise="medium",
                    forbidden=["techno", "rave", "nightclub"],
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # R04 — Techno vs Indie showdown                                      #
        # Strong opposing music preferences, needs a real compromise          #
        # ------------------------------------------------------------------ #
        {
            "group_id": "R04_techno_vs_indie",
            "description": "Techno fanatik vs indie ljubitelj — kompromis u muzici",
            "members": [
                _member(
                    "Vuk",
                    query="hoću techno ili dark electronic, nešto podzemno, dobar sound sistem, ne mainstream",
                    liked_types=["techno", "electronic", "club", "dark_electronic"],
                    noise="high",
                    earliest_hour=22,
                ),
                _member(
                    "Ana",
                    query="mrzim techno, volim indie folk ili akustičnu muziku, nešto intimno i toplo",
                    liked_types=["indie", "folk", "acoustic", "live_music", "singer_songwriter"],
                    noise="low",
                    forbidden=["techno", "rave", "nightclub", "electronic"],
                ),
                _member(
                    "Stefan",
                    query="ok sam i sa elektronskom i sa živom muzikom, bitno da je dobra atmosfera i dobra ekipa",
                    liked_types=["live_music", "electronic", "concert", "festival", "bar"],
                    noise="medium_high",
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # R05 — Deep Jazz night (Friday, late)                                #
        # Date-specific: Friday evening, jazz only                            #
        # ------------------------------------------------------------------ #
        {
            "group_id": "R05_jazz_petkom_uvece",
            "description": "Džez večer u petak — bebop, swing, moderni džez",
            "members": [
                _member(
                    "Vladislav",
                    query="petak uveče, tražim džez bar ili džez klub u Beogradu, volim bebop i swing",
                    liked_types=["jazz", "live_music", "bar", "bebop", "swing"],
                    noise="medium",
                    earliest_hour=20,
                    budget_max_rsd=2000,
                ),
                _member(
                    "Николь",
                    query="пятничный вечер, хочу послушать джаз, желательно живое выступление, не попса",
                    liked_types=["jazz", "live_music", "concert", "bar"],
                    noise="medium",
                    earliest_hour=20,
                    forbidden=["techno", "rave", "pop", "nightclub"],
                ),
                _member(
                    "Dragana",
                    query="džez je odličan, volim i soul i funk, nešto u intimnom prostoru, ne preveliki bend",
                    liked_types=["jazz", "soul", "funk", "live_music", "acoustic"],
                    noise="medium",
                    earliest_hour=19,
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # R06 — Classical music evening (specific date: February)             #
        # ------------------------------------------------------------------ #
        {
            "group_id": "R06_klasika_februar",
            "description": "Klasična muzika — koncerti u februaru, besplatno ako može",
            "members": [
                _member(
                    "Gordana",
                    query="tražim klasični koncert u februaru u Beogradu, besplatno bi bilo idealno",
                    liked_types=["classical_music", "concert", "philharmonic", "chamber_music"],
                    noise="low",
                    must_be_free=True,
                    forbidden=["nightclub", "rave", "techno", "pop"],
                ),
                _member(
                    "Михаил",
                    query="люблю классическую музыку, ищу концерт в феврале, лучше в Cultural Center или филармонии",
                    liked_types=["classical_music", "concert", "opera", "philharmonic"],
                    noise="low",
                    forbidden=["techno", "rave", "nightclub"],
                ),
                _member(
                    "Branka",
                    query="klasična muzika ili opera, ako nema onda nešto cultural i tiho, muzej ili izložba",
                    liked_types=["classical_music", "opera", "museum", "exhibition", "culture"],
                    noise="low",
                    forbidden=["nightclub", "rave", "techno"],
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # R07 — Savamala vibe seekers                                         #
        # Looking for the Savamala neighborhood: art, alternative, design     #
        # ------------------------------------------------------------------ #
        {
            "group_id": "R07_savamala_vibe",
            "description": "Savamala scena — art, alternativa, dizajn, kreativni prostor",
            "members": [
                _member(
                    "Jelena",
                    query="hoću u Savamalu, volim kreativne prostore, galerije, popup izložbe i alternativnu scenu",
                    liked_types=["gallery", "art", "alternative", "design", "creative_space"],
                    noise="medium",
                    forbidden=["mainstream", "nightclub"],
                ),
                _member(
                    "Ольга",
                    query="слышала про Saвамала, хочу посмотреть уличное искусство и зайти в интересные кафе или галереи",
                    liked_types=["street_art", "cafe", "gallery", "art", "design"],
                    noise="low",
                ),
                _member(
                    "Chris (Australia)",
                    query="I love creative districts with independent galleries, coffee shops and street art, Savamala sounds perfect",
                    liked_types=["gallery", "art", "cafe", "street_art", "design"],
                    noise="low",
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # R08 — Kalemegdan / Stari Grad outdoor vibe                          #
        # Scenic, historical, outdoor — sunset walks + cultural spots         #
        # ------------------------------------------------------------------ #
        {
            "group_id": "R08_kalemegdan_setnja",
            "description": "Kalemegdan i Stari Grad — šetnja, priroda, istorija, panorama",
            "members": [
                _member(
                    "Snežana",
                    query="hoću da se prošetam uz Kalemegdan ili Stari Grad, nešto kulturno napolju, lepe panorame",
                    liked_types=["outdoor", "park", "nature", "history", "culture"],
                    noise="low",
                    must_be_free=True,
                ),
                _member(
                    "Hiroshi (Japan)",
                    query="interested in historical sites, fortress walks, and open-air cultural events near Kalemegdan",
                    liked_types=["history", "outdoor", "culture", "exhibition", "architecture"],
                    noise="low",
                ),
                _member(
                    "Тамара",
                    query="люблю прогулки на природе, хочу посмотреть Калемегдан и окрестности, что-то на свежем воздухе",
                    liked_types=["outdoor", "nature", "park", "history", "culture"],
                    noise="low",
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # R09 — Rooftop / terrace summer vibe                                 #
        # Looking for terrace bars, open-air, summer atmosphere               #
        # ------------------------------------------------------------------ #
        {
            "group_id": "R09_rooftop_terasa",
            "description": "Letnja terasa ili rooftop bar — pogled, koktel, opuštena atmosfera",
            "members": [
                _member(
                    "Mina",
                    query="tražim rooftop bar ili lepu terasu u Beogradu, hoću pogled na grad i dobra pića",
                    liked_types=["bar", "rooftop", "terrace", "cocktail", "outdoor"],
                    noise="medium",
                    earliest_hour=18,
                    budget_max_rsd=3000,
                ),
                _member(
                    "Лена",
                    query="хочу провести вечер на открытой терасе, красивый вид, летняя атмосфера, коктейли",
                    liked_types=["terrace", "bar", "cocktail", "outdoor", "summer"],
                    noise="medium",
                    earliest_hour=18,
                ),
                _member(
                    "David (Germany)",
                    query="looking for a nice outdoor bar with a view, sunset drinks, relaxed vibe, not too loud",
                    liked_types=["outdoor", "bar", "terrace", "rooftop", "wine"],
                    noise="medium",
                    forbidden=["techno", "rave"],
                    earliest_hour=18,
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # R10 — Underground / alternative culture                             #
        # Subculture, underground clubs, experimental art                     #
        # ------------------------------------------------------------------ #
        {
            "group_id": "R10_underground_scena",
            "description": "Underground i alternativa — eksperimentalno, subkultura, nezavisna scena",
            "members": [
                _member(
                    "Relja",
                    query="hoću nešto underground, eksperimentalna muzika ili performans, van mejnstrima",
                    liked_types=["underground", "experimental", "alternative", "performance_art", "club"],
                    noise="high",
                    earliest_hour=21,
                ),
                _member(
                    "Арина",
                    query="интересует андеграундная сцена — экспериментальная музыка, независимые галереи, что-то нестандартное",
                    liked_types=["experimental", "underground", "art", "gallery", "alternative"],
                    noise="medium_high",
                    forbidden=["mainstream", "pop"],
                ),
                _member(
                    "Zara (France)",
                    query="I'm into alternative and experimental art scenes, looking for something off the beaten path in Belgrade",
                    liked_types=["experimental", "art", "alternative", "underground", "performance"],
                    noise="medium",
                    forbidden=["mainstream", "pop", "nightclub"],
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # R11 — Hip-hop / urban kultura                                       #
        # ------------------------------------------------------------------ #
        {
            "group_id": "R11_hiphop_urban",
            "description": "Hip-hop i urbana kultura — beat, rap, graffiti, street art",
            "members": [
                _member(
                    "Boban",
                    query="tražim hip-hop žurku ili rap koncert u Beogradu, ili neku street art izložbu",
                    liked_types=["hip_hop", "rap", "concert", "street_art", "urban"],
                    noise="high",
                    earliest_hour=21,
                ),
                _member(
                    "Кирилл",
                    query="люблю хип-хоп и граффити культуру, ищу концерт или событие связанное с уличным искусством",
                    liked_types=["hip_hop", "urban", "street_art", "graffiti", "concert"],
                    noise="high",
                    earliest_hour=20,
                ),
                _member(
                    "Lana",
                    query="volim urban kulturu ali ne mora biti samo hip-hop, i R&B i electronic su ok, bitna je energija",
                    liked_types=["hip_hop", "rnb", "electronic", "concert", "urban"],
                    noise="high",
                    earliest_hour=21,
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # R12 — Folk / etno muzika i srpska tradicija                         #
        # ------------------------------------------------------------------ #
        {
            "group_id": "R12_folk_etno_tradicija",
            "description": "Folk i etno — srpska tradicija, narodni program, starogradska muzika",
            "members": [
                _member(
                    "Dragan",
                    query="hoću da čujem srpsku narodnu muziku, kafana sa živom svirkom, starogradske pesme",
                    liked_types=["folk", "traditional", "live_music", "kafana", "serbian"],
                    noise="medium_high",
                    earliest_hour=19,
                ),
                _member(
                    "Светлана",
                    query="интересует сербская народная музыка и традиционная атмосфера, кафана с живой музыкой",
                    liked_types=["folk", "traditional", "live_music", "serbian", "culture"],
                    noise="medium",
                    earliest_hour=19,
                ),
                _member(
                    "Bosa",
                    query="volim etno muziku i world music, ali i srpski folk, nešto autentično i toplo",
                    liked_types=["folk", "etno", "world_music", "traditional", "live_music"],
                    noise="medium",
                    forbidden=["techno", "rave", "nightclub"],
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # R13 — Valentine's weekend date (EN + SR)                            #
        # Date-specific: romantic, February 14, special atmosphere            #
        # ------------------------------------------------------------------ #
        {
            "group_id": "R13_valentinovo_date",
            "description": "Valentinovo — romantičan izlazak, posebna atmosfera, februar",
            "members": [
                _member(
                    "Aleksandar",
                    query="planiramo romantičnu večeru za Valentinovo 14. februara, hoću nešto posebno sa lepim ambijentom",
                    liked_types=["restaurant", "wine", "romantic", "live_music", "acoustic"],
                    noise="low",
                    budget_max_rsd=6000,
                    earliest_hour=19,
                    forbidden=["nightclub", "rave", "techno"],
                ),
                _member(
                    "Sophie (France)",
                    query="Valentine's Day dinner, I want something romantic and atmospheric, good wine, maybe live acoustic music",
                    liked_types=["restaurant", "wine", "romantic", "acoustic", "culture"],
                    noise="low",
                    budget_max_rsd=5000,
                    earliest_hour=19,
                    forbidden=["nightclub", "rave", "techno"],
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # R14 — Letnji festival avgust (Summer festival, August)              #
        # Date-specific: August, outdoor festival, music + food + art         #
        # ------------------------------------------------------------------ #
        {
            "group_id": "R14_letnji_festival_avgust",
            "description": "Letnji festival u avgustu — muzika, hrana, umetnost napolju",
            "members": [
                _member(
                    "Petar",
                    query="tražim letnji festival u avgustu u Beogradu, muzika + hrana napolju, dan i noć",
                    liked_types=["festival", "outdoor", "live_music", "food", "summer"],
                    noise="high",
                ),
                _member(
                    "Masha (Russia)",
                    query="ищу летний фестиваль в августе в Белграде, музыка на открытом воздухе, еда и атмосфера",
                    liked_types=["festival", "outdoor", "live_music", "food", "summer"],
                    noise="high",
                ),
                _member(
                    "Luka",
                    query="festival je odličan, hoću i umetnost i film uz muziku, vikend program je ok",
                    liked_types=["festival", "film", "art", "live_music", "outdoor"],
                    noise="medium_high",
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # R15 — Nedeljno popodne (Sunday afternoon, relaxed)                  #
        # Specific vibe: Sunday, slow, family-adjacent, coffee culture        #
        # ------------------------------------------------------------------ #
        {
            "group_id": "R15_nedeljno_popodne",
            "description": "Nedeljno popodne — kafić, šetnja, izložba, nema žurke",
            "members": [
                _member(
                    "Iva",
                    query="nedeljno popodne, hoću nešto opušteno, kafić sa karakterom ili neka manja izložba",
                    liked_types=["cafe", "exhibition", "art", "outdoor", "design"],
                    noise="low",
                    forbidden=["nightclub", "rave", "techno"],
                ),
                _member(
                    "Вера",
                    query="воскресный день, хочу спокойно провести время — кофе, прогулка, небольшая выставка",
                    liked_types=["cafe", "exhibition", "outdoor", "art", "culture"],
                    noise="low",
                    forbidden=["nightclub", "rave"],
                ),
                _member(
                    "Ben (Netherlands)",
                    query="Sunday afternoon, looking for a nice coffee spot or a small exhibition, very relaxed day",
                    liked_types=["cafe", "art", "exhibition", "outdoor", "design"],
                    noise="low",
                    forbidden=["nightclub", "rave", "techno"],
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # R16 — Novi Beograd outdoor / Ušće park aktivnosti                   #
        # Location-specific: Novi Beograd, outdoor, sports, Ušće              #
        # ------------------------------------------------------------------ #
        {
            "group_id": "R16_novi_beograd_usće",
            "description": "Novi Beograd — Ušće park, sportske aktivnosti, napolju",
            "members": [
                _member(
                    "Nemanja",
                    query="živim u Novom Beogradu, hoću nešto blizu Ušća — sport, piknik, ili neka manifestacija napolju",
                    liked_types=["outdoor", "sports", "park", "nature", "picnic"],
                    noise="medium",
                    must_be_free=True,
                ),
                _member(
                    "Катерина",
                    query="ищу мероприятие на открытом воздухе в Белграде, парк или спорт, желательно бесплатно",
                    liked_types=["outdoor", "park", "sports", "nature", "free"],
                    noise="medium",
                    must_be_free=True,
                ),
                _member(
                    "Maja",
                    query="hoću nešto aktivno napolju, yoga u parku, biciklizam ili vikend festival na obali",
                    liked_types=["outdoor", "yoga", "sports", "cycling", "festival"],
                    noise="medium",
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # R17 — Acoustic / intimno, singer-songwriter evening                 #
        # Very specific vibe: acoustic, small venue, intimate                 #
        # ------------------------------------------------------------------ #
        {
            "group_id": "R17_akusticno_intimno",
            "description": "Akustična muzika — mali prostor, singer-songwriter, intiman nastup",
            "members": [
                _member(
                    "Tijana",
                    query="tražim akustični nastup, singer-songwriter ili akustični bend, mali prostor, intimna atmosfera",
                    liked_types=["acoustic", "singer_songwriter", "folk", "live_music", "indie"],
                    noise="low",
                    forbidden=["nightclub", "rave", "techno", "electronic"],
                    earliest_hour=19,
                ),
                _member(
                    "Элина",
                    query="хочу акустический концерт или живую музыку в небольшом пространстве, уютная атмосфера",
                    liked_types=["acoustic", "live_music", "singer_songwriter", "jazz", "folk"],
                    noise="low",
                    forbidden=["techno", "rave", "nightclub"],
                    earliest_hour=19,
                ),
                _member(
                    "Jonas (Sweden)",
                    query="looking for acoustic live music, a small intimate venue, singer-songwriter or quiet folk, not a big concert hall",
                    liked_types=["acoustic", "folk", "singer_songwriter", "indie", "live_music"],
                    noise="low",
                    forbidden=["techno", "rave", "nightclub"],
                    earliest_hour=19,
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # R18 — Potpuno tri različita muzička ukusa (3-way music compromise)   #
        # Electronic purist + classical lover + world music fan               #
        # ------------------------------------------------------------------ #
        {
            "group_id": "R18_tri_muzicka_ukusa",
            "description": "Tri muzička sveta — elektronika vs klasika vs world music",
            "members": [
                _member(
                    "Filip",
                    query="ja sam čisto za elektronsku muziku, dark techno ili ambient electronic, ne folk ne klasika",
                    liked_types=["electronic", "techno", "ambient", "club", "dark_electronic"],
                    noise="high",
                    forbidden=["classical_music", "folk", "opera"],
                    earliest_hour=22,
                ),
                _member(
                    "Анастасия",
                    query="я люблю классическую музыку и джаз, не понимаю электронику, хочу концерт или опус",
                    liked_types=["classical_music", "jazz", "opera", "concert", "philharmonic"],
                    noise="low",
                    forbidden=["techno", "rave", "electronic", "nightclub"],
                ),
                _member(
                    "Kofi (Ghana)",
                    query="I'm into world music and afrobeat, looking for something multicultural or with diverse music",
                    liked_types=["world_music", "afrobeat", "multicultural", "concert", "live_music"],
                    noise="medium_high",
                    forbidden=["classical_music"],
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # R19 — Vikend u mešanoj ekipi sa datumom (Saturday evening, budget)  #
        # Mixed group, Saturday specific, tight budget                        #
        # ------------------------------------------------------------------ #
        {
            "group_id": "R19_subota_vece_budzet",
            "description": "Subota uveče — mešana ekipa, budžet do 2000 RSD, nešto za svakoga",
            "members": [
                _member(
                    "Jovana",
                    query="subota uveče, budžet je mali max 2000 din, hoću bar ili koncert, neka živahna atmosfera",
                    liked_types=["bar", "concert", "live_music", "social"],
                    noise="medium_high",
                    budget_max_rsd=2000,
                    earliest_hour=20,
                ),
                _member(
                    "Pavle",
                    query="subota uveče, hoću nešto gde možemo svi da sedimo zajedno i pričamo, nije mi bitna muzika",
                    liked_types=["bar", "restaurant", "social", "cafe", "lounge"],
                    noise="medium",
                    budget_max_rsd=2000,
                    earliest_hour=19,
                ),
                _member(
                    "Анна",
                    query="субботний вечер, небольшой бюджет, хочу что-нибудь интересное — бар, концерт, или просто тусовка",
                    liked_types=["bar", "concert", "social", "party", "live_music"],
                    noise="medium",
                    budget_max_rsd=2000,
                    earliest_hour=20,
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # R20 — Kolege iz različitih zemalja (international team outing)      #
        # EN + RU + SR — after-conference dinner + drinks                     #
        # ------------------------------------------------------------------ #
        {
            "group_id": "R20_medjunarodna_ekipa",
            "description": "Međunarodna ekipa posle konferencije — večera + piće + konverzacija",
            "members": [
                _member(
                    "Elena (Serbia)",
                    query="organizujem izlazak za međunarodne kolege posle konferencije, dobar restoran pa piće",
                    liked_types=["restaurant", "bar", "wine", "social", "gastronomy"],
                    noise="medium",
                    earliest_hour=19,
                    forbidden=["nightclub", "rave", "techno"],
                ),
                _member(
                    "Michael (USA)",
                    query="after the conference, I want a nice dinner with local food and then maybe a cocktail bar, something authentic",
                    liked_types=["restaurant", "bar", "cocktail", "food", "local"],
                    noise="medium",
                    earliest_hour=19,
                    forbidden=["nightclub", "rave", "techno"],
                ),
                _member(
                    "Сергей",
                    query="после конференции хочу в хороший ресторан с сербской едой, потом в приятный бар с живой музыкой",
                    liked_types=["restaurant", "food", "bar", "live_music", "traditional"],
                    noise="medium",
                    earliest_hour=19,
                    forbidden=["techno", "rave"],
                ),
                _member(
                    "Liu Wei (China)",
                    query="looking for a nice restaurant with local cuisine and a quiet bar afterwards, good atmosphere for conversation",
                    liked_types=["restaurant", "bar", "food", "local", "social"],
                    noise="low",
                    earliest_hour=19,
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
    languages: list[str]
    candidates: list[tuple[NormalizedEvent, float]]
    ranked: list[tuple[NormalizedEvent, float, float, float]]
    latency_s: float
    hard_pass: int
    hard_total: int


def _detect_languages(members: list[dict]) -> list[str]:
    langs = set()
    _sr_words = {"hoću", "tražim", "volim", "imam", "nešto", "beogradu", "uveče", "hocu", "trazim", "nesto", "zelim", "nekoga", "volim", "hocemo"}
    _en_words = {"i", "want", "looking", "love", "like", "good", "please", "need", "interested", "open", "find", "something"}
    for m in members:
        q = m["query"]
        if any(ord(c) > 0x400 for c in q):
            langs.add("RU")
        q_lower = q.lower()
        words = set(q_lower.split())
        if words & _sr_words:
            langs.add("SR")
        if words & _en_words:
            langs.add("EN")
    return sorted(langs) if langs else ["?"]


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
        languages=_detect_languages(members),
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
    print("  BELGRADE GROUP RECOMMENDER — MULTILINGUAL TEST REPORT")
    print(_SEP)

    for res in results:
        langs = "+".join(res.languages) if res.languages else "?"
        print(f"\n{_SEP_THIN}")
        print(f"  {res.group_id}  [{langs}]  |  {res.description}")
        print(_SEP_THIN)
        print(f"  Members    : {', '.join(res.member_names)}")
        print(f"  Languages  : {langs}")
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
    print(f"  {'Group':<30} {'Lang':>8} {'Lat':>7}  {'Cands':>5}  {'Hard%':>6}  {'TopMin':>7}  {'TopMean':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*7}  {'-'*5}  {'-'*6}  {'-'*7}  {'-'*8}")
    for res in results:
        hard_pct = f"{100 * res.hard_pass / res.hard_total:.0f}%" if res.hard_total else "N/A"
        top_min = f"{res.ranked[0][2]:.4f}" if res.ranked else "N/A"
        top_mean = f"{res.ranked[0][3]:.4f}" if res.ranked else "N/A"
        langs = "+".join(res.languages) if res.languages else "?"
        print(
            f"  {res.group_id:<30} {langs:>8} {res.latency_s:>6.3f}s"
            f"  {len(res.candidates):>5}  {hard_pct:>6}  {top_min:>7}  {top_mean:>8}"
        )

    # Language breakdown
    print(f"\n{_SEP}")
    lang_groups: dict[str, list[GroupResult]] = {}
    for res in results:
        key = "+".join(res.languages)
        lang_groups.setdefault(key, []).append(res)
    print("  RESULTS BY LANGUAGE MIX")
    print(_SEP_THIN)
    for lang, grps in sorted(lang_groups.items()):
        scored = [r for r in grps if r.ranked]
        avg_min = sum(r.ranked[0][2] for r in scored) / len(scored) if scored else 0
        avg_cands = sum(len(r.candidates) for r in grps) / len(grps)
        print(f"  {lang:<12} {len(grps):>2} groups  avg_candidates={avg_cands:.1f}  avg_top_min={avg_min:.4f}")
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
    log.info("Step 4/4 — Running %d multilingual test groups …", len(groups))
    results: list[GroupResult] = []
    for group in groups:
        log.info("  %s …", group["group_id"])
        try:
            res = run_group(group, indexable, corpus_matrix, encoder)
            results.append(res)
            langs = "+".join(res.languages)
            log.info(
                "  Done [%s] — %d candidates, %d ranked, %.3fs",
                langs, len(res.candidates), len(res.ranked), res.latency_s,
            )
        except Exception:
            log.exception("  %s FAILED", group["group_id"])

    print_report(results)


if __name__ == "__main__":
    main()
