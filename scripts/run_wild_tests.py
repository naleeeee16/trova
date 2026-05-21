#!/usr/bin/env python3
"""
Wild / extreme scenario test suite — 21 groups with bizarre, conflicting, and hyper-specific
constraints designed to stress-test the recommender's ranking logic.

Categories:
  W01-W03  Extreme ambience conflicts
  W04-W06  Bizarre hard constraints
  W07-W10  Subcultures & hobbies
  W11-W13  Gastronomic extremes
  W14-W17  Social experiments
  W18-W20  Sensory battles
  W21      Original: The Night Owl Intellectuals (own scenario)

Usage:
    python run_wild_tests.py --skip-ingest
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
log = logging.getLogger("wildtest")

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
# CATEGORY 1: Extreme ambience conflicts
# ---------------------------------------------------------------------------

def build_test_groups() -> list[dict]:
    return [

        # ------------------------------------------------------------------ #
        # W01 — The Vampire & The Plant Lover                                 #
        # Darkness obsessive vs sunlight-only garden lover                    #
        # Expected: 0 candidates — hard semantic conflict                     #
        # ------------------------------------------------------------------ #
        {
            "group_id": "W01_vampire_vs_plant_lover",
            "description": "Vampir traži podrum bez svetla, botaničar traži baštu sa suncem — nemoguć kompromis",
            "members": [
                _member(
                    "Vlad (fotofobija)",
                    query="tražim isključivo zatvorene podzemne prostore, bez prozora, bez prirodnog svetla, mračni basement barovi ili podrumski klubovi u Beogradu",
                    liked_types=["basement", "underground", "dark", "club", "cave_bar"],
                    noise="medium",
                    forbidden=["outdoor", "garden", "terrace", "open_air", "park", "nature"],
                    earliest_hour=20,
                ),
                _member(
                    "Flora (botaničarka)",
                    query="hoću isključivo bašte, terase ili prostori sa zelenilom i prirodnom svetlošću, sunce je obavezno, bez podruma",
                    liked_types=["garden", "outdoor", "terrace", "nature", "park", "botanical"],
                    noise="low",
                    forbidden=["basement", "underground", "nightclub", "rave", "dark"],
                    must_be_free=True,
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # W02 — The Silent Raver                                              #
        # Techno max noise vs migraine absolute silence — same group          #
        # Expected: 0 candidates                                              #
        # ------------------------------------------------------------------ #
        {
            "group_id": "W02_silent_raver",
            "description": "Techno manijak + migrena u istoj grupi — nema kompromisa",
            "members": [
                _member(
                    "Dušan (techno)",
                    query="hoću najglasniji techno klub u Beogradu, što glasniji sub-bass, dark techno, industrial, brutalni sound sistem",
                    liked_types=["techno", "industrial", "rave", "dark_techno", "club"],
                    noise="high",
                    earliest_hour=23,
                ),
                _member(
                    "Tijana (migrena)",
                    query="imam migrenu i ne mogu da podnesem nikakvu buku, tražim apsolutnu tišinu, možda meditacija ili tiha izložba, nikako muzika",
                    liked_types=["meditation", "exhibition", "yoga", "silence", "reading"],
                    noise="low",
                    forbidden=["techno", "rave", "nightclub", "concert", "festival", "bar", "party"],
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # W03 — The Digital Detox Influencer                                  #
        # No-phone zone vs must-have Instagram aesthetic                      #
        # ------------------------------------------------------------------ #
        {
            "group_id": "W03_detox_vs_influencer",
            "description": "Digital detoks vs Instagram influencer — offline prostor vs fotografisanje svega",
            "members": [
                _member(
                    "Jovan (digital detox)",
                    query="tražim mesto gde su telefoni zabranjeni ili bar gde nema wi-fi, priroda i odsustvo tehnologije, digitalni detoks, offline iskustvo u Beogradu",
                    liked_types=["nature", "outdoor", "meditation", "detox", "silent_retreat", "analog"],
                    noise="low",
                    forbidden=["tech", "gaming", "nightclub", "rave"],
                ),
                _member(
                    "Mia (influencer)",
                    query="tražim najfotogeničnije mesto u Beogradu, Instagram zidovi, neonske instalacije, estetski interijer, savršen lighting za fotke, place to be seen",
                    liked_types=["aesthetic", "design", "art_installation", "cafe", "gallery", "neon"],
                    noise="medium",
                    forbidden=["outdoor", "park", "nature"],
                    budget_max_rsd=3000,
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # CATEGORY 2: Bizarre practical hard constraints                      #
        # ------------------------------------------------------------------ #

        # ------------------------------------------------------------------ #
        # W04 — The Great Dane & The Cyclist                                  #
        # Giant dog + expensive bike = very specific venue needs              #
        # ------------------------------------------------------------------ #
        {
            "group_id": "W04_pas_i_bicikl",
            "description": "Nemačka doga + skupi bicikl — pet friendly + bike parking unutra ili vidno mesto",
            "members": [
                _member(
                    "Olja (sa nemačkom dogom)",
                    query="dolazim sa ogromnim psom — nemačka doga, treba mi pet friendly kafić ili bašta u Beogradu gde mogu da uđem sa velikim psom, ne mali prostor",
                    liked_types=["pet_friendly", "outdoor", "garden", "cafe", "park", "dog_friendly"],
                    noise="medium",
                    forbidden=["nightclub", "rave", "underground", "basement"],
                ),
                _member(
                    "Marko (na biciklu)",
                    query="dolazim skupim biciklom koji ne mogu da ostavim napolju, tražim kafić ili bar gde mogu da uvezem bicikl unutra ili ga park na vidnom mestu ispred ulaza",
                    liked_types=["outdoor", "cafe", "bar", "bike_friendly", "terrace"],
                    noise="medium",
                    forbidden=["nightclub", "underground"],
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # W05 — The Allergy Minefield                                         #
        # Nuts + gluten + alcohol + smoke — impossible Belgrade night out     #
        # ------------------------------------------------------------------ #
        {
            "group_id": "W05_alergije_minefield",
            "description": "Alergija na orašaste + celijakija + ne pije alkohol + bez dima — Beograd nightmare",
            "members": [
                _member(
                    "Petra (alergije)",
                    query="imam alergiju na orašaste plodove i celijakiju, tražim restoran ili kafić bez glutena i bez orašastih, mora biti čisto i sigurno za alergičare",
                    liked_types=["vegan", "gluten_free", "health_food", "cafe", "restaurant"],
                    noise="low",
                    forbidden=["nightclub", "rave", "bar", "pub"],
                ),
                _member(
                    "Nikola (ne pije)",
                    query="ne pijem alkohol i ne podnosim dim, tražim smoke-free prostor u Beogradu sa bezalkoholnim opcijama, kafić ili kulturni prostor",
                    liked_types=["cafe", "juice_bar", "culture", "exhibition", "non_alcoholic"],
                    noise="low",
                    forbidden=["nightclub", "rave", "bar", "pub", "smoking"],
                ),
                _member(
                    "Анна (веган без глютена)",
                    query="веган без глютена, не переношу табачный дым, ищу кафе или ресторан в Белграде с веганским меню и чистым воздухом",
                    liked_types=["vegan", "gluten_free", "cafe", "restaurant", "organic"],
                    noise="low",
                    forbidden=["nightclub", "rave", "bar", "smoke"],
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # W06 — The 5 AM Junction                                             #
        # Night shift worker winding down + marathon runner waking up        #
        # ------------------------------------------------------------------ #
        {
            "group_id": "W06_pet_sati_ujutru",
            "description": "Noćna smena (sprema se za spavanje) + maratonac (sprema se za trening) — u 5 ujutru",
            "members": [
                _member(
                    "Bojan (noćna smena)",
                    query="završio sam smenu u 5 ujutru, hoću nešto tiho gde mogu da sedim uz čaj ili kafu i opustim se, ne hoću gungulu, možda 24h kafić",
                    liked_types=["cafe", "tea", "quiet", "24h", "chill", "breakfast"],
                    noise="low",
                    budget_max_rsd=500,
                    earliest_hour=5,
                    forbidden=["nightclub", "rave", "concert", "party"],
                ),
                _member(
                    "Tamara (maratonac)",
                    query="vstajem u 5 ujutru pre treninga, tražim mesto gde mogu da jedem proteinski doručak ili ovsenu kašu, energetski obrok, zdravo, brzo",
                    liked_types=["breakfast", "healthy_food", "cafe", "sports_nutrition", "early_morning"],
                    noise="low",
                    budget_max_rsd=800,
                    earliest_hour=5,
                    forbidden=["nightclub", "rave", "bar"],
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # CATEGORY 3: Subcultures & hobbies                                  #
        # ------------------------------------------------------------------ #

        # ------------------------------------------------------------------ #
        # W07 — The Yugo-Brutalist Geeks                                      #
        # Events specifically IN brutalist buildings — Genex, Sava Centar    #
        # ------------------------------------------------------------------ #
        {
            "group_id": "W07_brutalisticki_geeks",
            "description": "Traže događaje isključivo u brutalističkim zgradama — Genex, Sava Centar, Blokovi",
            "members": [
                _member(
                    "Rajko (arhitektura)",
                    query="obožavam jugoslovensku brutalističku arhitekturu, tražim događaje koji se organizuju u Genex tornji, Sava Centru, Palati Albanija ili nekim blokovima Novog Beograda",
                    liked_types=["architecture", "brutalism", "yugo_heritage", "culture", "history"],
                    noise="low",
                ),
                _member(
                    "Sasha (urban explorer)",
                    query="ищу события в брутальных социалистических зданиях Белграда — Genex, Sava Centar, бетонные блоки Нового Белграда, советская эстетика",
                    liked_types=["brutalism", "architecture", "urban_exploration", "history", "soviet_aesthetic"],
                    noise="low",
                ),
                _member(
                    "Ivana (fotografkinja)",
                    query="fotografkinja, hoću da fotografišem brutalističke enterijere dok se tamo nešto organizuje, izložba, koncert ili predavanje u betonskim prostorima",
                    liked_types=["photography", "architecture", "exhibition", "brutalism", "culture"],
                    noise="low",
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # W08 — The Solo-Together Readers                                     #
        # Four people who want to be in the same cafe but NOT talk to each   #
        # ------------------------------------------------------------------ #
        {
            "group_id": "W08_solo_together_citaci",
            "description": "Četvoro prijatelja koji zajedno izlaze ali ne smeju da pričaju — čitanje u tišini",
            "members": [
                _member(
                    "Jelena (čita roman)",
                    query="tražim kafić u Beogradu koji je poznat po tome da ljudi rade, uče ili čitaju u njemu, tiho, ni biblioteka ni klub — možda čitaonica ili quiet cafe",
                    liked_types=["cafe", "library", "reading", "quiet", "study_cafe", "book"],
                    noise="low",
                    forbidden=["nightclub", "rave", "concert", "party", "karaoke"],
                ),
                _member(
                    "Petar (radi od doma)",
                    query="tražim coworking ili kafić za rad, bez buke, dobar wi-fi, mogu da sedim satima, ne smetaju mi drugi ljudi koji rade u tišini",
                    liked_types=["cafe", "coworking", "quiet", "work_friendly", "study"],
                    noise="low",
                    forbidden=["nightclub", "rave", "concert", "bar"],
                ),
                _member(
                    "Mira (studentkinja)",
                    query="hoću kafić gde nema gungule, gde se uči, quiet zone, ne može da se priča glasno, tražim tiho mesto za učenje uz kafu",
                    liked_types=["cafe", "quiet", "study", "library", "academic"],
                    noise="low",
                    forbidden=["nightclub", "rave", "concert", "party"],
                ),
                _member(
                    "Lenya (чтение)",
                    query="ищу тихое кафе в Белграде для чтения — без музыки, без разговоров, люблю библиотечную атмосферу",
                    liked_types=["cafe", "library", "quiet", "reading", "book"],
                    noise="low",
                    forbidden=["nightclub", "rave", "bar", "party"],
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # W09 — The Occult & Goth Night                                       #
        # Crystal shops, tarot, gothic industrial, dark ambient               #
        # ------------------------------------------------------------------ #
        {
            "group_id": "W09_okultna_goth_noc",
            "description": "Okultna prodavnica, tarot predavanje ili gothic ambient/dark industrial zvuk",
            "members": [
                _member(
                    "Morrigan",
                    query="tražim prodavnice kristala, tarot predavanja ili ezoteričke radionice u Beogradu, sve što je mračno i mistično, alternativna scena",
                    liked_types=["occult", "tarot", "crystal", "esoteric", "workshop", "alternative"],
                    noise="low",
                    forbidden=["mainstream", "pop", "nightclub"],
                ),
                _member(
                    "Viktor (gothic)",
                    query="tražim gothic industrial ili dark ambient koncert ili klub u Beogradu, mračna estetika, crno oblačenje, ne mainstream techno",
                    liked_types=["gothic", "industrial", "dark_ambient", "alternative", "concert"],
                    noise="medium_high",
                    forbidden=["mainstream", "pop", "top40"],
                    earliest_hour=21,
                ),
                _member(
                    "Lilith",
                    query="gothic, dark aesthetics, black metal ili dark wave muzika, estetika vampira ili okultnog, traži mi se nešto autentično u Beogradu",
                    liked_types=["dark_wave", "black_metal", "gothic", "occult", "concert"],
                    noise="medium_high",
                    forbidden=["mainstream", "pop", "techno"],
                    earliest_hour=20,
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # W10 — The Anime & Cosplay Hub                                       #
        # Costume-ok venues, bubble tea, manga collection                     #
        # ------------------------------------------------------------------ #
        {
            "group_id": "W10_anime_cosplay",
            "description": "Anime & cosplay — traže mesta gde se prihvataju kostimi, bubble tea i manga",
            "members": [
                _member(
                    "Akemi (cosplay)",
                    query="tražim mesto u Beogradu gde mogu da dođem u anime kostimu, gde se prodaje bubble tea i ima manga biblioteka ili anime screaning",
                    liked_types=["anime", "cosplay", "gaming", "manga", "pop_culture", "bubble_tea"],
                    noise="medium",
                    forbidden=["formal", "nightclub"],
                ),
                _member(
                    "Darko (otaku)",
                    query="otaku sam, tražim gaming cafe ili manga bar u Beogradu, hoću da vidim ljude koji se interesuju za anime i japansku pop kulturu",
                    liked_types=["gaming", "anime", "manga", "cafe", "japanese_culture"],
                    noise="medium",
                    budget_max_rsd=1500,
                ),
                _member(
                    "Yui (iz Japana)",
                    query="I'm from Japan, looking for an anime cafe, gaming lounge or any event about Japanese pop culture in Belgrade",
                    liked_types=["anime", "japanese_culture", "gaming", "pop_culture", "manga"],
                    noise="medium",
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # CATEGORY 4: Gastronomic extremes                                    #
        # ------------------------------------------------------------------ #

        # ------------------------------------------------------------------ #
        # W11 — The Carnivore vs. Raw Vegan                                   #
        # All-you-can-eat meat vs raw unprocessed vegan only                  #
        # ------------------------------------------------------------------ #
        {
            "group_id": "W11_mesar_vs_sirovi_vegan",
            "description": "All-you-can-eat meso vs sirova neobrađena veganska hrana — gastronomski rat",
            "members": [
                _member(
                    "Bogdan (mesojed)",
                    query="hoću all-you-can-eat meso, roštilj, pljeskavice, janjetina, dobar srpski grill, što više mesa što bolje, ugostiti celu ekipu",
                    liked_types=["bbq", "grill", "meat", "restaurant", "traditional", "buffet"],
                    noise="medium_high",
                    forbidden=["vegan", "vegetarian"],
                ),
                _member(
                    "Zoe (raw vegan)",
                    query="ja sam raw vegan, jedem isključivo sirovu neobrađenu biljnu hranu, bez termičke obrade, bez glutena, bez šećera, tražim raw vegan restoran ili cafe u Beogradu",
                    liked_types=["raw_vegan", "vegan", "health_food", "organic", "sprouts"],
                    noise="low",
                    forbidden=["meat", "grill", "bbq", "restaurant"],
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # W12 — The Secret Speakeasy Hunt                                     #
        # Bars with no sign, hidden entrances, phone booth doors              #
        # ------------------------------------------------------------------ #
        {
            "group_id": "W12_tajni_speakeasy",
            "description": "Traže skrivene barove bez natpisa, ulaz kroz telefonsku govornicu ili skriveni prolaz",
            "members": [
                _member(
                    "Nikola (cocktail geek)",
                    query="tražim speakeasy barove u Beogradu, mesta bez natpisa na ulazu, skriveni barovi, ulaz kroz telefonsku govornicu ili frizerski salon, visoki koktel craft mixology",
                    liked_types=["speakeasy", "cocktail", "bar", "hidden_bar", "craft_cocktail", "mixology"],
                    noise="medium",
                    budget_max_rsd=3000,
                    earliest_hour=20,
                ),
                _member(
                    "Zara (mystery lover)",
                    query="I want to find hidden bars in Belgrade, the kind you'd never find without a recommendation, mysterious entrances, no sign on the door, good cocktails",
                    liked_types=["hidden_bar", "speakeasy", "cocktail", "underground", "bar"],
                    noise="medium",
                    budget_max_rsd=3000,
                    earliest_hour=20,
                    forbidden=["mainstream", "tourist"],
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # W13 — The Craft Beer vs Rakija Battle                               #
        # 50 craft IPAs vs grandma's homemade šljiva in a čokanj             #
        # ------------------------------------------------------------------ #
        {
            "group_id": "W13_craft_beer_vs_rakija",
            "description": "50 vrsta craft piva vs domaća šljiva u čokanju — pivski geek vs srpski patriota",
            "members": [
                _member(
                    "Luke (craft beer nerd)",
                    query="tražim craft beer bar u Beogradu sa barem 20-50 vrsta piva, IPA, stout, sour ale, imperial, tap list koji se menja, ne komercijalnog piva",
                    liked_types=["craft_beer", "ipa", "stout", "bar", "beer_tasting", "tap"],
                    noise="medium",
                    budget_max_rsd=2500,
                    forbidden=["wine", "cocktail"],
                ),
                _member(
                    "Miodrag (rakijaš)",
                    query="hoću domaću rakiju, šljivovicu iz sela, dobar izbor domaćih rakija, kafana sa starinom, nijedna pivnica nije to što tražim, srpska tradicija",
                    liked_types=["rakija", "traditional", "kafana", "serbian", "folk", "spirits"],
                    noise="medium_high",
                    budget_max_rsd=1500,
                    forbidden=["craft_beer", "cocktail", "wine_bar"],
                ),
                _member(
                    "Jovana (neutralna)",
                    query="ok sam i sa pivom i sa rakijom, bitno da je dobra atmosfera, mogu da pijem i sok, samo hoću zabavno veče",
                    liked_types=["bar", "social", "live_music", "traditional", "local"],
                    noise="medium",
                    budget_max_rsd=2000,
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # CATEGORY 5: Social experiments                                      #
        # ------------------------------------------------------------------ #

        # ------------------------------------------------------------------ #
        # W14 — The Ex-Couple Compromise                                      #
        # Two exes in the same group, need a big enough space to avoid each  #
        # ------------------------------------------------------------------ #
        {
            "group_id": "W14_bivsi_par_kompromis",
            "description": "Bivši partneri u istoj grupi — hoće da budu u istom prostoru ali ne smeju da se sretnu",
            "members": [
                _member(
                    "Stefan (bivši dečko)",
                    query="dolazim na grupni izlazak ali ima tu i moja bivša devojka, tražim nešto veliko napolju ili festival gde ima mnogo prostora i gužve, ne mali intiman prostor",
                    liked_types=["festival", "outdoor", "park", "large_venue", "street_event"],
                    noise="medium_high",
                ),
                _member(
                    "Mila (bivša devojka)",
                    query="imam malo neprijatnu situaciju, dolazi moj bivši, hoću da budemo u istom prostoru ali na rastojanju, veliko mesto, festival ili park, ne mali bar",
                    liked_types=["festival", "outdoor", "park", "large_venue", "market"],
                    noise="medium",
                    forbidden=["small_bar", "intimate_venue"],
                ),
                _member(
                    "Ivan (zajednički drug)",
                    query="organizujem izlazak, ima tu komplicirane situacije između prijatelja, hoću nešto opušteno i prostorno gde svi mogu da se snađu na svoju ruku",
                    liked_types=["festival", "outdoor", "market", "food_festival", "park"],
                    noise="medium",
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # W15 — The Anti-Mainstream Hipsters                                  #
        # Forbidden: Beograd na Vodi, mainstream clubs; required: vinyl only  #
        # ------------------------------------------------------------------ #
        {
            "group_id": "W15_anti_mainstream_hipsters",
            "description": "Zabranjuju BnV i svaki mainstream — traže isključivo vinyl barove i underground",
            "members": [
                _member(
                    "Rex (vinyl DJ)",
                    query="zabranjeno sve što je Beograd na Vodi, zabranjeno sve što je mainstream, tražim isključivo mesta koja puštaju vinyl ploče, underground lokali, mali, autentični, nekomercijalni barovi",
                    liked_types=["vinyl", "record_bar", "underground", "alternative", "independent", "jazz", "soul"],
                    noise="medium",
                    forbidden=["mainstream", "nightclub", "tourist", "commercial", "top40"],
                    earliest_hour=20,
                ),
                _member(
                    "Lola (indie kurator)",
                    query="tražim mesta koja se ne pojavljuju u turističkim vodičima, lokalna skrivena mesta, nezavisna kulturna scena, anti-komercijalno, vinyl muzika preferred",
                    liked_types=["independent", "alternative", "vinyl", "bar", "culture", "underground"],
                    noise="medium",
                    forbidden=["tourist", "mainstream", "commercial"],
                    earliest_hour=19,
                ),
                _member(
                    "Saša (record collector)",
                    query="kolekcionujem vinyl, tražim vintage prodavnice, record fair ili bar sa dobrom vinyl kolekcijom, soul, funk, jazz, ska",
                    liked_types=["vinyl", "record_store", "jazz", "soul", "funk", "ska", "alternative"],
                    noise="medium",
                    budget_max_rsd=2000,
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # W16 — The Survivalists                                              #
        # Bushcraft, wilderness skills, foraging workshops around Belgrade    #
        # ------------------------------------------------------------------ #
        {
            "group_id": "W16_survivalists",
            "description": "Traže veštine preživljavanja, bushcraft, foraging oko Beograda",
            "members": [
                _member(
                    "Bear (survivalist)",
                    query="tražim bushcraft ili wilderness survival radionice oko Beograda, vatrogašenje, pravljenje zaklona, branje divljih biljaka, foraging, kampovanje bez opreme",
                    liked_types=["bushcraft", "survival", "outdoor", "nature", "workshop", "foraging"],
                    noise="low",
                    forbidden=["nightclub", "rave", "urban"],
                ),
                _member(
                    "Ana (foragerkinja)",
                    query="zanima me branje divljih biljaka i gljiva oko Beograda, wild food tour, herbalism, botanika u prirodi",
                    liked_types=["foraging", "nature", "outdoor", "hiking", "botany", "workshop"],
                    noise="low",
                    forbidden=["nightclub", "rave", "city"],
                ),
                _member(
                    "Сергей (кемпинг)",
                    query="ищу кемпинг или выживальческий воркшоп около Белграда, навыки выживания в дикой природе, без гаджетов",
                    liked_types=["camping", "survival", "nature", "outdoor", "workshop"],
                    noise="low",
                    forbidden=["nightclub", "rave", "urban", "city"],
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # W17 — The Boardgame Strategists                                     #
        # Need 100+ titles, specifically Catan tournaments, no casual gaming  #
        # ------------------------------------------------------------------ #
        {
            "group_id": "W17_boardgame_stratezi",
            "description": "Traže kafić sa 100+ naslova, organizovani Catan turnir, ozbiljni stratezi",
            "members": [
                _member(
                    "Filip (Catan master)",
                    query="tražim board game kafić u Beogradu koji ima Catan, Ticket to Ride, Pandemic i slične strateške igre, idealno organizovani turniri, ne džepne igre",
                    liked_types=["boardgame", "strategy_game", "cafe", "gaming", "catan", "tournament"],
                    noise="medium",
                    forbidden=["nightclub", "rave", "concert"],
                    budget_max_rsd=1500,
                ),
                _member(
                    "Ema (puzzle strategist)",
                    query="obožavam board games, RPG, strateške igre, tražim dedicated board game bar ili kafić u Beogradu sa velikom kolekcijom",
                    liked_types=["boardgame", "rpg", "cafe", "gaming", "strategy_game"],
                    noise="medium",
                    budget_max_rsd=1500,
                    forbidden=["nightclub", "rave"],
                ),
                _member(
                    "Chris (D&D player)",
                    query="looking for a tabletop RPG venue or board game cafe in Belgrade where I can play D&D or strategy games for an entire evening",
                    liked_types=["boardgame", "tabletop_rpg", "dnd", "gaming", "cafe"],
                    noise="medium",
                    budget_max_rsd=2000,
                    forbidden=["nightclub", "rave", "techno"],
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # CATEGORY: Sensory battles                                           #
        # ------------------------------------------------------------------ #

        # ------------------------------------------------------------------ #
        # W18 — The Perfume Sensitive Squad                                   #
        # Neutral scent environment: no incense, no heavy food, no smoke     #
        # ------------------------------------------------------------------ #
        {
            "group_id": "W18_osjetljivi_na_miris",
            "description": "Apsolutno neutralan miris — bez tamjana, bez dima, bez masne hrane",
            "members": [
                _member(
                    "Vera (alergija na mirise)",
                    query="tražim mesta bez mirišljavih štapića, bez jakih parfema, bez dima, bez masnih mirisa hrane, open-air ili dobro ventilisani prostor, minimalistički ambijent",
                    liked_types=["outdoor", "minimalist", "garden", "cafe", "fresh_air"],
                    noise="low",
                    forbidden=["nightclub", "rave", "smoking", "incense"],
                ),
                _member(
                    "Max (migren i mirisi)",
                    query="migrene mi okidaju mirisi, tražim outdoor terasu ili kafić sa neutralnim mirisom, bez parfema i bez dima, neutralno okruženje",
                    liked_types=["outdoor", "terrace", "fresh_air", "cafe", "park"],
                    noise="low",
                    forbidden=["nightclub", "rave", "smoking", "bar"],
                ),
                _member(
                    "Юля (чувствительность)",
                    query="у меня сильная чувствительность к запахам, ищу кафе или место на свежем воздухе без дыма и ароматических свечей",
                    liked_types=["outdoor", "cafe", "fresh_air", "terrace", "garden"],
                    noise="low",
                    forbidden=["nightclub", "smoking", "bar"],
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # W19 — The River vs. Concrete Battle                                 #
        # Seasick person (no splavs) vs water view obsessive                 #
        # ------------------------------------------------------------------ #
        {
            "group_id": "W19_reka_vs_beton",
            "description": "Morska bolest (bez splavova) vs opsesija pogledom na vodu — nemoguć kompromis",
            "members": [
                _member(
                    "Mirko (morska bolest)",
                    query="dobijem morsku bolest na splavovima i plovilima, ne mogu na vodu uopšte, tražim nešto u centru, beton i kopno, nije bitno da ima vodu",
                    liked_types=["cafe", "bar", "gallery", "restaurant", "urban", "city_center"],
                    noise="medium",
                    forbidden=["splav", "floating_bar", "boat", "river", "waterfront"],
                ),
                _member(
                    "Nina (water obsession)",
                    query="hoću isključivo pogled na Savu ili Dunav, splav ili kafić pored reke, bez pogleda na vodu ne mogu ni da razmišljam o izlasku",
                    liked_types=["splav", "river_view", "waterfront", "outdoor", "terrace"],
                    noise="medium",
                    forbidden=["basement", "underground", "no_view"],
                ),
                _member(
                    "Ivo (miritelj)",
                    query="pokušavam da pomognem grupi, ok sam sa svim, hoću samo dobru atmosferu i ok hranu",
                    liked_types=["restaurant", "bar", "outdoor", "social", "food"],
                    noise="medium",
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # W20 — The Total Introvert Social                                    #
        # In the crowd, but zero eye contact — dark cinema, silent bars      #
        # ------------------------------------------------------------------ #
        {
            "group_id": "W20_introvert_social",
            "description": "Hoće gužvu ali nula kontakta očima — tamni bioskopi sa hranom, tihi barovi",
            "members": [
                _member(
                    "Lena (introvert)",
                    query="hoću da budem okružena ljudima ali ne hoću da niko priča sa mnom, tamni bioskop sa hranom ili tihi bar gde se ne očekuje interakcija, parallel play energy",
                    liked_types=["cinema", "dark_bar", "film", "quiet_bar", "solo_friendly"],
                    noise="low",
                    forbidden=["karaoke", "quiz_night", "networking", "speed_dating", "interactive"],
                ),
                _member(
                    "Тим (социофобия)",
                    query="социофоб ищет место где можно быть среди людей но без контакта — кино с едой, тихий бар, не нетворкинг",
                    liked_types=["cinema", "film", "quiet", "dark_bar", "solo"],
                    noise="low",
                    forbidden=["networking", "karaoke", "quiz", "interactive", "party"],
                ),
                _member(
                    "Hugo (parallel play)",
                    query="I love being around people but hate talking to them, looking for a cinema with food service, a quiet bar or a dark venue where everyone minds their own business",
                    liked_types=["cinema", "film", "dark_bar", "quiet_venue", "solo_friendly"],
                    noise="low",
                    forbidden=["karaoke", "quiz_night", "networking", "interactive", "party"],
                ),
            ],
        },

        # ------------------------------------------------------------------ #
        # W21 — The Night Owl Intellectuals (original scenario)               #
        # People who debate philosophy until 4 AM — need late-night quiet    #
        # venue with drinks, no music pressure, maybe books on shelves        #
        # ------------------------------------------------------------------ #
        {
            "group_id": "W21_nocni_sofisti",
            "description": "[ORIGINAL] Noćni filozofi — debata do 4 ujutru, tiho piće, police sa knjigama, bez muzike",
            "members": [
                _member(
                    "Aleksej (filosof)",
                    query="hoćemo da pijemo i razgovaramo o filozofiji do 4 ujutru, tražim kafić ili bar koji radi kasno, tiho, gde muzika nije previše glasna i možemo slobodno da pričamo satima",
                    liked_types=["late_night", "bar", "cafe", "quiet", "book_bar", "intellectual"],
                    noise="medium",
                    budget_max_rsd=2000,
                    earliest_hour=22,
                    forbidden=["techno", "rave", "nightclub", "karaoke"],
                ),
                _member(
                    "Иван (философ, русский)",
                    query="хочу провести ночь за философскими разговорами в баре с книгами на полках, тихая атмосфера, открыто до поздна, не клуб",
                    liked_types=["bar", "book_bar", "quiet", "late_night", "intellectual", "cafe"],
                    noise="medium",
                    budget_max_rsd=1500,
                    earliest_hour=22,
                    forbidden=["techno", "rave", "nightclub"],
                ),
                _member(
                    "Claire (philosopher, EN)",
                    query="we want to debate and talk for hours over wine or beer, a late-night bar with a quiet vibe, maybe bookshelves, no DJ, no dance floor",
                    liked_types=["bar", "wine", "quiet", "late_night", "book", "intellectual", "cafe"],
                    noise="medium",
                    budget_max_rsd=2000,
                    earliest_hour=22,
                    forbidden=["techno", "rave", "nightclub", "karaoke"],
                ),
                _member(
                    "Milena (knjige i vino)",
                    query="tražim kafanu ili bar koji ima knjige, vino, tih ambijent i radi do kasno — filozofski talk do zore",
                    liked_types=["wine_bar", "book_bar", "cafe", "quiet", "late_night", "social"],
                    noise="medium",
                    budget_max_rsd=2000,
                    earliest_hour=22,
                    forbidden=["techno", "rave", "nightclub"],
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
    expected_zero: bool  # groups designed to return 0 results


_EXPECTED_ZERO = {"W01_vampire_vs_plant_lover", "W02_silent_raver"}


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
        expected_zero=group["group_id"] in _EXPECTED_ZERO,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
_SEP = "=" * 72
_SEP_THIN = "-" * 72


def print_report(results: list[GroupResult]) -> None:
    print(f"\n{_SEP}")
    print("  BELGRADE GROUP RECOMMENDER — WILD / EXTREME SCENARIO REPORT")
    print(_SEP)

    for res in results:
        zero_label = "  [EXPECTED: 0 results — impossible constraints]" if res.expected_zero else ""
        print(f"\n{_SEP_THIN}")
        print(f"  {res.group_id}  |  {res.description}")
        if zero_label:
            print(f" {zero_label}")
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
            if res.expected_zero:
                print("\n  PASS — Correctly returned 0 results for impossible constraint combination.")
            else:
                print("\n  WARNING: No results returned (dataset may not cover this niche).")

    # Summary
    print(f"\n{_SEP}")
    print("  SUMMARY")
    print(_SEP)
    print(f"  {'Group':<32} {'Lat':>7}  {'Cands':>5}  {'Hard%':>6}  {'TopMin':>7}  {'Status':>12}")
    print(f"  {'-'*32} {'-'*7}  {'-'*5}  {'-'*6}  {'-'*7}  {'-'*12}")
    for res in results:
        hard_pct = f"{100 * res.hard_pass / res.hard_total:.0f}%" if res.hard_total else "—"
        top_min = f"{res.ranked[0][2]:.4f}" if res.ranked else "—"
        if res.expected_zero and not res.ranked:
            status = "PASS (0 exp)"
        elif not res.ranked:
            status = "WARN (0 res)"
        elif res.hard_pass == res.hard_total:
            status = "OK"
        else:
            status = "WARN (hard)"
        print(
            f"  {res.group_id:<32} {res.latency_s:>6.3f}s"
            f"  {len(res.candidates):>5}  {hard_pct:>6}  {top_min:>7}  {status:>12}"
        )

    # Category analysis
    print(f"\n{_SEP}")
    print("  CONFLICT ANALYSIS — groups with < 5 candidates")
    print(_SEP_THIN)
    sparse = [r for r in results if len(r.candidates) < 5]
    if sparse:
        for res in sparse:
            verdict = "expected" if res.expected_zero else "dataset gap or overly specific"
            print(f"  {res.group_id:<32}  {len(res.candidates)} candidates  [{verdict}]")
    else:
        print("  All groups returned 5+ candidates.")
    print(_SEP)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    log.info("Step 4/4 — Running %d wild scenario groups …", len(groups))
    results: list[GroupResult] = []
    for group in groups:
        log.info("  %s …", group["group_id"])
        try:
            res = run_group(group, indexable, corpus_matrix, encoder)
            results.append(res)
            note = " [EXPECTED 0]" if (res.expected_zero and not res.ranked) else ""
            log.info(
                "  Done — %d candidates, %d ranked, %.3fs%s",
                len(res.candidates), len(res.ranked), res.latency_s, note,
            )
        except Exception:
            log.exception("  %s FAILED", group["group_id"])

    print_report(results)


if __name__ == "__main__":
    main()
