"""Build EventAttributes from a NormalizedEvent (Phase 2, rules + optional lazy Gemini)."""

from __future__ import annotations

import os

import belgrade_recommender.extract.rules as text_rules
from belgrade_recommender.ingest.models import NormalizedEvent
from belgrade_recommender.schemas.event_attributes import EventAttributes, NoiseHint

# Each hashtag (lowercased, # stripped) maps to 1-3 type_hint tokens.
# Tokens are the same snake_case vocabulary used in parsed_preferences liked_types.
_TAG_TO_TYPES: dict[str, list[str]] = {
    # ---- Live music --------------------------------------------------------
    "livemusic":              ["live_music"],
    "liveperformance":        ["live_music", "concert"],
    "livemusicfestival":      ["live_music", "festival"],
    "concert":                ["concert", "live_music"],
    "freeconcert":            ["concert", "live_music"],
    "music":                  ["live_music"],
    "musicevent":             ["live_music"],
    "musicfestival":          ["festival", "live_music"],
    "performance":            ["concert", "live_music"],
    "performanceart":         ["concert", "exhibition"],
    "openmic":                ["open_mic", "live_music"],
    "jamsession":             ["open_mic", "live_music"],
    # ---- Genre tags --------------------------------------------------------
    "jazz":                   ["jazz", "live_music"],
    "jazzfestival":           ["jazz", "festival", "live_music"],
    "jazzclub":               ["jazz", "live_music"],
    "jazznight":              ["jazz", "live_music", "bar"],
    "swing":                  ["jazz", "live_music"],
    "bebop":                  ["jazz", "live_music"],
    "blues":                  ["blues", "live_music"],
    "bluesmusic":             ["blues", "live_music"],
    "classicalmusic":         ["classical_music", "concert"],
    "classicalmusicconcert":  ["classical_music", "concert"],
    "opera":                  ["classical_music", "concert"],
    "philharmonic":           ["philharmonic", "classical_music", "concert"],
    "symphony":               ["classical_music", "concert"],
    "chambermusic":           ["chamber_music", "classical_music", "concert"],
    "rock":                   ["concert", "live_music"],
    "rockmusic":              ["concert", "live_music"],
    "indiemusic":             ["concert", "live_music"],
    "indierock":              ["concert", "live_music"],
    "alternativemusic":       ["concert", "live_music"],
    "popmusic":               ["concert", "live_music"],
    "folk":                   ["folk_traditional", "live_music"],
    "folkmusic":              ["folk_traditional", "live_music"],
    "balkanmusic":            ["folk_traditional", "live_music"],
    "serbianmusic":           ["folk_traditional", "live_music"],
    "traditionalmusic":       ["folk_traditional", "live_music"],
    "traditionalmusicnight":  ["folk_traditional", "live_music"],
    "ethnicmusic":            ["folk_traditional", "live_music"],
    "folkrock":               ["folk_traditional", "concert", "live_music"],
    "folktronica":            ["folk_traditional", "electronic", "live_music"],
    "neofolk":                ["folk_traditional", "live_music"],
    "indiefolk":              ["folk_traditional", "live_music"],
    "balkanfolk":             ["folk_traditional", "live_music"],
    "folklore":               ["folk_traditional", "culture"],
    "folkart":                ["folk_traditional", "exhibition"],
    "folkloreart":            ["folk_traditional", "exhibition"],
    "worldmusic":             ["world_music", "live_music"],
    "afrobeat":               ["world_music", "live_music"],
    "reggae":                 ["world_music", "live_music"],
    "soul":                   ["soul", "live_music"],
    "rnb":                    ["soul", "live_music"],
    "hiphop":                 ["hip_hop", "concert"],
    "hiphopfestival":         ["hip_hop", "festival", "live_music"],
    "rap":                    ["hip_hop", "concert"],
    "electronic":             ["electronic", "nightlife"],
    "electronicmusic":        ["electronic", "nightlife"],
    "techno":                 ["techno", "nightlife"],
    "darktechno":             ["dark_techno", "techno", "nightlife"],
    "darkelectronic":         ["dark_ambient", "nightlife"],
    "darkambient":            ["dark_ambient", "nightlife"],
    "industrial":             ["dark_ambient", "nightlife"],
    "industrialmusic":        ["dark_ambient", "nightlife"],
    "ebm":                    ["dark_ambient", "nightlife"],
    "darkwave":               ["dark_ambient", "nightlife"],
    "gothic":                 ["gothic", "dark_ambient"],
    "goth":                   ["gothic", "dark_ambient"],
    "gothicmusic":            ["gothic", "dark_ambient", "nightlife"],
    "metal":                  ["concert", "live_music"],
    "heavymetal":             ["concert", "live_music"],
    "avantgarde":             ["concert", "live_music"],
    "experimental":           ["concert", "live_music"],
    "experimentalmusic":      ["concert", "live_music"],
    "acoustic":               ["acoustic", "live_music"],
    "acousticmusic":          ["acoustic", "live_music"],
    "singersongwriter":       ["singer_songwriter", "acoustic", "live_music"],
    "djset":                  ["electronic", "nightlife"],
    "dj":                     ["electronic", "nightlife"],
    "vinyl":                  ["vinyl_music", "bar"],
    "vinylmarket":            ["vinyl_music", "market"],
    # ---- Nightlife / club --------------------------------------------------
    "nightlife":              ["nightlife"],
    "party":                  ["party", "nightlife"],
    "clubbing":               ["nightlife", "club"],
    "nightclub":              ["nightlife", "club"],
    "rave":                   ["rave", "nightlife", "techno"],
    "openairrave":            ["rave", "nightlife", "techno"],
    "openairparty":           ["nightlife", "party", "rave"],
    "clubnight":              ["nightlife", "club"],
    # ---- Adult-content events ----------------------------------------------
    "adultsonly":             ["adults_only"],
    "fetish":                 ["adults_only", "nightlife"],
    "playparty":              ["adults_only", "nightlife", "party"],
    "kinkybelgrade":          ["adults_only", "nightlife"],
    "kinkparty":              ["adults_only", "nightlife", "party"],
    "kinkevent":              ["adults_only", "nightlife"],
    "bdsm":                   ["adults_only", "nightlife"],
    # ---- Art / exhibition --------------------------------------------------
    "artexhibition":          ["exhibition"],
    "exhibition":             ["exhibition"],
    "exhibit":                ["exhibition"],
    "contemporaryart":        ["exhibition"],
    "modernart":              ["exhibition"],
    "visualarts":             ["exhibition"],
    "art":                    ["exhibition"],
    "artinstallation":        ["exhibition"],
    "artworkshop":            ["workshop", "exhibition"],
    "sculpture":              ["exhibition"],
    "painting":               ["exhibition"],
    "illustration":           ["exhibition"],
    "comics":                 ["exhibition"],
    "digitalart":             ["exhibition"],
    "streetart":              ["street_art", "exhibition"],
    "publicart":              ["street_art", "exhibition"],
    "urbanart":               ["street_art"],
    "graffiti":               ["street_art"],
    "photography":            ["photography", "exhibition"],
    "photographyexhibition":  ["photography", "exhibition"],
    "naturephotography":      ["photography", "outdoor"],
    "travelphotography":      ["photography"],
    "design":                 ["design", "exhibition"],
    "interiordesign":         ["design"],
    "architecture":           ["architecture"],
    "culture":                ["culture"],
    "culturalevent":          ["culture"],
    "culturalheritage":       ["culture", "history"],
    "culturalexchange":       ["culture", "social"],
    "serbianculture":         ["culture"],
    "localculture":           ["culture"],
    # ---- Museum / history --------------------------------------------------
    "museum":                 ["museum"],
    "museumevent":            ["museum"],
    "museumexhibition":       ["museum", "exhibition"],
    "history":                ["history"],
    "historicaltour":         ["guided_tour", "history"],
    "historicsite":           ["history", "outdoor"],
    "heritage":               ["history", "culture"],
    # ---- Film / cinema -----------------------------------------------------
    "film":                   ["film"],
    "filmfestival":           ["film", "festival"],
    "filmscreening":          ["film"],
    "moviescreening":         ["film"],
    "cinema":                 ["film"],
    "documentaryfilm":        ["film"],
    "documentary":            ["film"],
    "indiefilm":              ["film"],
    "animation":              ["film"],
    "shortfilm":              ["film"],
    # ---- Theatre / comedy --------------------------------------------------
    "theatre":                ["theatre"],
    "theater":                ["theatre"],
    "drama":                  ["theatre"],
    "performingarts":         ["theatre", "concert"],
    "standupcomedy":          ["comedy"],
    "comedyshow":             ["comedy"],
    "comedy":                 ["comedy"],
    "improvtheatre":          ["comedy", "theatre"],
    # ---- Food & drink -------------------------------------------------------
    "foodfestival":           ["food_festival", "food"],
    "gastronomy":             ["food"],
    "foodie":                 ["food"],
    "food":                   ["food"],
    "foodanddrink":           ["food"],
    "foodandbeverage":        ["food"],
    "foodtour":               ["food", "guided_tour"],
    "foodevent":              ["food"],
    "restaurant":             ["restaurant"],
    "restaurantopening":      ["restaurant"],
    "finedining":             ["restaurant"],
    "restaurantreview":       ["restaurant"],
    "streetfood":             ["food", "outdoor"],
    "farmersmarket":          ["food", "market", "outdoor"],
    "nightmarket":            ["market", "outdoor"],
    "localfood":              ["food", "restaurant"],
    "localproduce":           ["food"],
    "winetasting":            ["wine", "bar"],
    "winefestival":           ["wine", "festival"],
    "cocktails":              ["cocktail_bar", "bar"],
    "cocktailevent":          ["cocktail_bar", "bar"],
    "craftbeer":              ["craft_beer", "bar"],
    "beertasting":            ["craft_beer", "bar"],
    "specialtycoffee":        ["cafe"],
    "coffeeshop":             ["cafe"],
    "coffeelover":            ["cafe"],
    "coffee":                 ["cafe"],
    "coffeeculture":          ["cafe"],
    "cafeculture":            ["cafe"],
    "cafeopening":            ["cafe"],
    "bakery":                 ["food", "cafe"],
    "dessert":                ["food", "cafe"],
    "breakfast":              ["breakfast", "food", "cafe"],
    "morningcoffee":          ["breakfast", "cafe"],
    "asiancuisine":           ["asian_cuisine", "restaurant"],
    "internationalcuisine":   ["restaurant"],
    "serbianfood":            ["serbian_cuisine", "restaurant"],
    "srpskahrana":            ["serbian_cuisine", "restaurant"],
    "serbiancuisine":         ["serbian_cuisine", "restaurant"],
    "balkancuisine":          ["serbian_cuisine", "restaurant"],
    "balkanculture":          ["folk_traditional", "culture"],
    "serbianculture":         ["culture", "folk_traditional"],
    "traditionalcuisine":     ["food", "serbian_cuisine", "restaurant"],
    "traditionalfood":        ["food", "folk_traditional"],
    "traditionaldish":        ["food", "serbian_cuisine"],
    "brunch":                 ["breakfast", "food", "cafe"],
    "lunch":                  ["lunch", "restaurant", "food"],
    "lunchspecial":           ["lunch", "restaurant"],
    "lunchmenu":              ["lunch", "restaurant"],
    "businesslunch":          ["lunch", "restaurant"],
    "dinner":                 ["dinner", "restaurant", "food"],
    "dinnerparty":            ["dinner", "restaurant"],
    "dinnershow":             ["dinner", "restaurant", "live_music"],
    "afterwork":              ["dinner", "restaurant", "bar"],
    "snack":                  ["snack", "food", "cafe"],
    # ---- Bar / cafe --------------------------------------------------------
    "bar":                    ["bar"],
    "cafe":                   ["cafe"],
    "radio":                  ["cafe", "bar"],
    "speakeasy":              ["speakeasy", "bar"],
    "rooftop":                ["rooftop_bar", "bar"],
    "rooftopbar":             ["rooftop_bar", "bar"],
    "cocktailbar":            ["cocktail_bar", "bar"],
    "pubevent":               ["bar"],
    "pubnight":               ["bar"],
    "kafana":                 ["kafana", "folk_traditional"],
    "kafananight":            ["kafana", "folk_traditional", "nightlife"],
    "rakija":                 ["folk_traditional", "kafana"],
    "serbianrakija":          ["folk_traditional"],
    "lounge":                 ["lounge", "bar"],
    "latenigthbar":           ["late_night_bar", "bar"],
    "latenightbar":           ["late_night_bar", "bar"],
    "latenight":              ["late_night_bar", "bar"],
    "bookbar":                ["bar", "lecture"],
    "splav":                  ["splav", "nightlife", "bar"],
    "splavovi":               ["splav", "nightlife"],
    "riverbar":               ["splav", "bar"],
    "riverview":              ["waterfront"],
    "waterfront":             ["waterfront"],
    "riverside":              ["waterfront", "outdoor"],
    # ---- Outdoor / nature --------------------------------------------------
    "hiking":                 ["hiking", "outdoor"],
    "hikingtrip":             ["hiking", "outdoor"],
    "hikingadventure":        ["hiking", "outdoor"],
    "hikeadventure":          ["hiking", "outdoor"],
    "excursion":              ["excursion", "outdoor"],
    "outdoor":                ["outdoor"],
    "outdooradventure":       ["outdoor", "hiking"],
    "outdooractivities":      ["outdoor"],
    "outdoorevent":           ["outdoor"],
    "outdoorrecreation":      ["outdoor"],
    "outdooractivity":        ["outdoor"],
    "nature":                 ["outdoor", "nature"],
    "naturewalk":             ["outdoor", "hiking"],
    "hikingtrail":            ["hiking", "outdoor"],
    "trekking":               ["hiking", "outdoor"],
    "daytrip":                ["excursion", "outdoor"],
    "weekendgetaway":         ["excursion", "outdoor"],
    "openair":                ["outdoor"],
    "ecotourism":             ["outdoor", "excursion"],
    "watersports":            ["outdoor", "sport"],
    "kayaking":               ["outdoor", "sport"],
    "sports":                 ["sport", "outdoor"],
    "fitness":                ["sport"],
    "wellness":               ["sport"],
    "yoga":                   ["yoga", "outdoor", "sport"],
    "running":                ["outdoor", "sport"],
    "cycling":                ["outdoor", "sport"],
    "adventure":              ["outdoor", "excursion"],
    "adventuretravel":        ["outdoor", "excursion"],
    "wildlife":               ["outdoor", "nature"],
    "camping":                ["outdoor", "nature"],
    "bushcraft":              ["bushcraft", "outdoor", "nature"],
    "survival":               ["outdoor", "nature"],
    "foraging":               ["outdoor", "nature"],
    "wildcrafting":           ["outdoor", "nature"],
    "botany":                 ["outdoor", "nature"],
    "urban_gardening":        ["outdoor", "nature"],
    "dance":                  ["dance", "nightlife"],
    "dancing":                ["dance", "nightlife"],
    "salsa":                  ["dance", "nightlife"],
    "meditation":             ["meditation"],
    "meditacija":             ["meditation"],
    "grill":                  ["grill", "restaurant"],
    "rostilj":                ["grill", "restaurant"],
    "bbq":                    ["grill", "restaurant"],
    # ---- Family ------------------------------------------------------------
    "familyevent":            ["family_friendly"],
    "familyfun":              ["family_friendly"],
    "familyfriendly":         ["family_friendly"],
    "kidsactivities":         ["family_friendly"],
    "kidsevent":              ["family_friendly"],
    "eastercarnival":         ["family_friendly", "festival"],
    # ---- Education / lecture -----------------------------------------------
    "lecture":                ["lecture"],
    "workshop":               ["workshop"],
    "workshops":              ["workshop"],
    "masterclass":            ["workshop"],
    "education":              ["lecture"],
    "science":                ["lecture"],
    "astronomy":              ["lecture"],
    "publiclecture":          ["lecture"],
    "publictalk":             ["lecture"],
    "literaryevent":          ["lecture"],
    "literature":             ["lecture"],
    "literatureclub":         ["lecture", "social"],
    "bookdiscussion":         ["lecture", "social"],
    "bookclub":               ["lecture", "social"],
    "bookfair":               ["lecture", "market"],
    "bookpresentation":       ["lecture"],
    "readingprogram":         ["lecture"],
    "discussion":             ["lecture"],
    "tech":                   ["tech_meetup"],
    "technology":             ["tech_meetup"],
    "startup":                ["tech_meetup", "networking"],
    "devfest":                ["tech_meetup", "lecture"],
    "hackathon":              ["tech_meetup", "workshop"],
    "techconference":         ["tech_meetup", "lecture"],
    "techevent":              ["tech_meetup"],
    "businessforum":          ["networking", "lecture"],
    "entrepreneurship":       ["networking", "lecture"],
    # ---- Social / networking -----------------------------------------------
    "networking":             ["networking"],
    "networkingevent":        ["networking"],
    "teambuilding":           ["networking", "social"],
    "communityevent":         ["social"],
    "communitygathering":     ["social"],
    "communityengagement":    ["social"],
    "socialgathering":        ["social"],
    "languagelearning":       ["social", "language_exchange"],
    "languageexchange":       ["language_exchange", "social"],
    "expat":                  ["language_exchange", "social"],
    "socialcommentary":       ["social"],
    "gaming":                 ["gaming"],
    "anime":                  ["anime", "gaming"],
    "manga":                  ["anime"],
    "cosplay":                ["cosplay", "anime"],
    "animeevent":             ["anime"],
    "bubbletea":              ["cafe", "asian_cuisine"],
    "kpop":                   ["kpop"],
    "quiznight":              ["quiz", "bar"],
    "trivia":                 ["quiz", "bar"],
    "trivianight":            ["quiz", "bar"],
    "boardgame":              ["board_game", "cafe"],
    "boardgames":             ["board_game", "cafe"],
    "tabletop":               ["board_game"],
    "tabletopgaming":         ["board_game"],
    "tabletoproleplay":       ["board_game"],
    "dnd":                    ["board_game"],
    "rpg":                    ["board_game", "gaming"],
    "strategygame":           ["board_game"],
    # ---- Esoteric / spiritual -----------------------------------------------
    "tarot":                  ["esoteric"],
    "occult":                 ["esoteric"],
    "esoteric":               ["esoteric"],
    "crystal":                ["esoteric"],
    "crystals":               ["esoteric"],
    "spirituality":           ["esoteric", "meditation"],
    "pagan":                  ["esoteric"],
    "witchcraft":             ["esoteric"],
    "astrology":              ["esoteric"],
    # ---- Dietary / lifestyle -----------------------------------------------
    "vegan":                  ["vegan"],
    "veganfood":              ["vegan", "food"],
    "veganevent":             ["vegan"],
    "veganfestival":          ["vegan", "food_festival", "festival"],
    "veganoption":            ["vegan"],
    "vegetarian":             ["vegan"],
    "vegetarianfood":         ["vegan", "food"],
    "plantbased":             ["vegan"],
    "glutenfree":             ["gluten_free"],
    "dairyfree":              ["gluten_free"],
    "organic":                ["organic", "food"],
    "organicfood":            ["organic", "food"],
    "rawvegan":               ["raw_vegan", "vegan"],
    "rawfood":                ["raw_vegan"],
    "nonalcoholic":           ["non_alcoholic", "bar"],
    "mocktail":               ["non_alcoholic", "bar"],
    "mocktails":              ["non_alcoholic", "bar"],
    "zeroalcohol":            ["non_alcoholic"],
    # ---- Fashion / vintage -------------------------------------------------
    "thrift":                 ["market"],
    "thriftstore":            ["market"],
    "vintageclothing":        ["market"],
    "vintagefashion":         ["market"],
    "streetfashion":          ["fashion"],
    "fashionshow":            ["fashion"],
    "fashion":                ["fashion"],
    # ---- Market / craft ----------------------------------------------------
    "craftfair":              ["craft_fair", "market"],
    "craftmarket":            ["craft_fair", "market"],
    "handmade":               ["market"],
    "handmademarket":         ["market"],
    "market":                 ["market"],
    "swapmeet":               ["market"],
    "fleamarket":             ["market"],
    "audioequipment":         ["vinyl_music", "market"],
    # ---- Guided tours ------------------------------------------------------
    "guidedtour":             ["guided_tour"],
    "grouptour":              ["guided_tour"],
    "citytour":               ["guided_tour"],
    "cityexploration":        ["guided_tour", "outdoor"],
    "urbanexploration":       ["outdoor"],
    # ---- Hidden / local ----------------------------------------------------
    "hiddengems":             ["hidden_gem"],
    "localbusiness":          ["social"],
    # ---- Festival (general) ------------------------------------------------
    "festival":               ["festival"],
    "summerfestival":         ["festival", "outdoor"],
    "summerevent":            ["festival"],
    "winterfestival":         ["festival"],
    "celebration":            ["party", "celebration"],
    "newopening":             ["social"],
    "grandopening":           ["social"],
}

_NOISY_TYPE_HINTS = frozenset({
    "techno", "dark_techno", "electronic", "nightlife", "party", "club",
    "hip_hop", "dark_ambient", "gothic", "rave",
})
_QUIET_TYPE_HINTS = frozenset({
    "lecture", "exhibition", "museum", "classical_music",
    "workshop", "film", "theatre", "esoteric", "meditation",
})
_MEDIUM_HIGH_HINTS = frozenset({
    "live_music", "concert", "jazz", "blues", "folk_traditional",
    "open_mic", "world_music", "soul", "singer_songwriter",
})

_OUTDOOR_TAGS = frozenset({
    "hiking", "outdoor", "excursion", "nature", "openair",
    "trekking", "outdooradventure", "outdooractivities",
    "watersports", "kayaking", "camping",
})
_INDOORISH_TAGS = frozenset({
    "museum", "exhibition", "lecture", "cinema", "theatre", "theater",
})


def _type_hints_from_tags(tags: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in tags:
        key = raw.lower().replace("#", "").replace(" ", "").replace("-", "")
        hints = _TAG_TO_TYPES.get(key, [])
        for hint in hints:
            if hint not in seen:
                seen.add(hint)
                out.append(hint)
    return out


def _noise_hint_from_types(type_hints: list[str], tags: list[str]) -> NoiseHint:
    types_set = set(type_hints)
    if types_set & _NOISY_TYPE_HINTS:
        return "high"
    if types_set & _QUIET_TYPE_HINTS:
        return "low"
    if types_set & _MEDIUM_HIGH_HINTS:
        return "medium_high"
    # Fall back to raw tag scan for tags not yet in the mapping
    lowered = {t.lower().replace("#", "") for t in tags}
    if lowered & {"techno", "rave", "openair", "nightlife", "party", "livemusic"}:
        return "high"
    if lowered & {"lecture", "exhibition", "museum", "classicalmusic", "workshop"}:
        return "low"
    return "unknown"


def _outdoor_hint(tags: list[str], text_lower: str) -> bool | None:
    lowered = {t.lower().replace("#", "") for t in tags}
    if lowered & _OUTDOOR_TAGS or "outdoor" in text_lower or "open-air" in text_lower or "napolju" in text_lower:
        return True
    if lowered & _INDOORISH_TAGS and not (lowered & _OUTDOOR_TAGS):
        return False
    return None


# How many characters to look back for negation signals before a keyword match.
_NEGATION_WINDOW = 45
_NEGATION_SIGNALS = (
    "not ", "non-", "without ", "after a ", "after the ", "for non",
    "isn't ", "no ", "except ", "instead of ", "skip ", "despite ",
    "unlike ", "leaving ", "avoiding ",
)

# Keywords whose type_hint should only be added when they appear in a positive
# (non-negated) context.  A phrase like "after a vegan post" or "not vegan"
# would otherwise inject a false dietary type_hint onto the event.
_CONTEXT_SENSITIVE_KW: frozenset[str] = frozenset({
    "vegan", "plant-based", "gluten-free", "gluten free",
})


def _kw_positive_context(text_lower: str, kw: str) -> bool:
    """Return True only when *kw* appears somewhere not preceded by a negation signal."""
    idx = 0
    while True:
        pos = text_lower.find(kw, idx)
        if pos == -1:
            break
        window = text_lower[max(0, pos - _NEGATION_WINDOW): pos]
        if not any(neg in window for neg in _NEGATION_SIGNALS):
            return True
        idx = pos + 1
    return False


def extract_event_attributes(event: NormalizedEvent) -> EventAttributes:
    text = (event.event_description_resolved or "") + "\n" + (event.event_description_raw or "")
    text_lower = text.lower()

    free = text_rules.detect_free_entry(text)
    if not free:
        _free_tag_signals = {"freeevent", "freeadmission", "freeentry", "freeconcert",
                             "freeconcerts", "freetickets", "freeaccess", "noadmission"}
        _tag_set = {t.lower().replace("#", "").replace(" ", "").replace("-", "") for t in event.tags}
        free = bool(_tag_set & _free_tag_signals)
    rsd = text_rules.max_rsd_amount(text)
    eur = text_rules.max_eur_amount(text)
    paid_hint = text_rules.paid_language_hint(text) and not free

    type_hints = _type_hints_from_tags(event.tags)

    # Keyword scan of description to catch what hashtags miss
    _kw_map = [
        ("jazz",             "jazz"),
        ("bebop",            "jazz"),
        ("swing music",      "jazz"),
        ("swing jazz",       "jazz"),
        ("swing dance",      "jazz"),
        ("philharmonic",     "philharmonic"),
        ("symphony",         "classical_music"),
        ("classical music",  "classical_music"),
        ("chamber music",    "chamber_music"),
        ("hiking",           "hiking"),
        ("exhibition",       "exhibition"),
        ("lecture",          "lecture"),
        ("workshop",         "workshop"),
        ("concert",          "concert"),
        ("restaurant",       "restaurant"),
        ("photography",      "photography"),
        ("gallery",          "exhibition"),
        ("festival",         "festival"),
        ("opera",            "classical_music"),
        ("film",             "film"),
        ("cinema",           "film"),
        ("comedy",           "comedy"),
        ("theatre",          "theatre"),
        ("theater",          "theatre"),
        ("networking",       "networking"),
        ("startup",          "tech_meetup"),
        ("hackathon",        "tech_meetup"),
        ("craft beer",       "craft_beer"),
        ("wine",             "wine"),
        ("cocktail",         "cocktail_bar"),
        ("rooftop",          "rooftop_bar"),
        ("speakeasy",        "speakeasy"),
        ("vinyl",            "vinyl_music"),
        ("open mic",         "open_mic"),
        ("open-mic",         "open_mic"),
        ("family",           "family_friendly"),
        ("kids",             "family_friendly"),
        ("children",         "family_friendly"),
        ("street art",       "street_art"),
        ("graffiti",         "street_art"),
        ("kafana",           "kafana"),
        ("folk_traditional", "folk_traditional"),
        ("grill",            "grill"),
        ("rostilj",          "grill"),
        # NOTE: "dance" intentionally REMOVED from _kw_map — substring match is too greedy:
        # "guidance", "abundance", "endurance" all contain "dance" and cause false positives.
        # Dance events are covered by #dance and #salsa in _TAG_TO_TYPES instead.
        ("yoga",             "yoga"),
        ("meditation",       "meditation"),
        ("meditacija",       "meditation"),
        ("lounge",           "lounge"),
        ("language exchange","language_exchange"),
        ("asian cuisine",    "asian_cuisine"),
        ("sushi",            "asian_cuisine"),
        ("ramen",            "asian_cuisine"),
        ("serbian cuisine",  "serbian_cuisine"),
        ("srpska kuhinja",   "serbian_cuisine"),
        ("rakija",           "folk_traditional"),
        ("turbofolk",        "folk_traditional"),
        ("turbo folk",       "folk_traditional"),
        ("turbo-folk",       "folk_traditional"),
        ("narodna muzika",   "folk_traditional"),
        ("srpska muzika",    "folk_traditional"),
        ("balkan music",     "folk_traditional"),
        ("balkan muzika",    "folk_traditional"),
        ("hip hop",          "hip_hop"),
        ("hip-hop",          "hip_hop"),
        ("rap music",        "hip_hop"),
        ("vegan",            "vegan"),
        ("plant-based",      "vegan"),
        ("gluten-free",      "gluten_free"),
        ("gluten free",      "gluten_free"),
        ("tarot",            "esoteric"),
        ("occult",           "esoteric"),
        # Genre keywords missing from hashtags (e.g. "rave" in title, "techno" in body)
        ("rave",             "rave"),
        ("techno",           "techno"),
        ("gabber",           "techno"),
        ("acid techno",      "techno"),
        ("hard techno",      "techno"),
        ("hard groove",      "techno"),
        ("happy hardcore",   "techno"),
        # Adult-content keywords — ensures filtering even without explicit hashtags
        ("kink party",       "adults_only"),
        ("kink event",       "adults_only"),
        ("fetish",           "adults_only"),
        ("shibari",          "adults_only"),
        ("wax play",         "adults_only"),
        ("adults only",      "adults_only"),
        ("adults-only",      "adults_only"),
        ("18+ only",         "adults_only"),
        ("strictly 18",      "adults_only"),
        ("entry strictly 18", "adults_only"),
        ("вход строго 18",   "adults_only"),
        ("splav",            "splav"),
        ("board game",       "board_game"),
        ("board games",      "board_game"),
        ("tabletop",         "board_game"),
        ("bubble tea",       "asian_cuisine"),
        ("cosplay",          "cosplay"),
        ("book club",        "social"),
        ("brutalism",        "architecture"),
        ("architectural tour", "architecture"),
        ("manga",            "anime"),
        ("breakfast",        "breakfast"),
        ("brunch",           "breakfast"),
        ("dorucak",          "breakfast"),
        ("doručak",          "breakfast"),
        ("завтрак",          "breakfast"),
        ("lunch",            "lunch"),
        ("rucak",            "lunch"),
        ("ručak",            "lunch"),
        ("обед",             "lunch"),
        ("dinner",           "dinner"),
        ("vecera",           "dinner"),
        ("večera",           "dinner"),
        ("ужин",             "dinner"),
        ("after work",       "dinner"),
        ("afterwork",        "dinner"),
        ("snack",            "snack"),
        ("grickalice",       "snack"),
        ("grickalica",       "snack"),
    ]
    seen = set(type_hints)
    for kw, label in _kw_map:
        if label in seen:
            continue
        if kw in _CONTEXT_SENSITIVE_KW:
            # Only add dietary/lifestyle type_hints when the keyword appears
            # in a clearly affirmative context — not in negated phrases like
            # "after a vegan post" or "not gluten-free".
            if _kw_positive_context(text_lower, kw):
                type_hints.append(label)
                seen.add(label)
        elif kw in text_lower:
            type_hints.append(label)
            seen.add(label)

    noise = _noise_hint_from_types(type_hints, event.tags)

    return EventAttributes(
        event_id=event.event_id,
        price_free_signal=free,
        price_paid_signal=paid_hint or bool(rsd or eur),
        price_amount_rsd=rsd,
        price_amount_eur=eur,
        city_hint=text_rules.detect_city_hint(text),
        date_snippets=text_rules.extract_date_snippets(text),
        day_of_week_hints=text_rules.extract_day_of_week(text),
        type_hints=type_hints,
        noise_level_hint=noise,
        outdoor_hint=_outdoor_hint(event.tags, text_lower),
        extraction_method="rules_v2",
    )


def _lazy_gemini_env_enabled() -> bool:
    return os.environ.get("USE_LAZY_GEMINI_EVENT_ATTRIBUTES", "").strip().lower() in (
        "1", "true", "yes",
    )


def _merge_rules_with_gemini_lazy(base: EventAttributes, gem: EventAttributes) -> EventAttributes:
    """Keep rule-based price/city signals; enrich noise, outdoor, types, dates from Gemini when useful."""
    noise = gem.noise_level_hint if gem.noise_level_hint != "unknown" else base.noise_level_hint
    outdoor = gem.outdoor_hint if gem.outdoor_hint is not None else base.outdoor_hint
    city = gem.city_hint if base.city_hint == "unknown" and gem.city_hint != "unknown" else base.city_hint
    type_hints = list(dict.fromkeys([*base.type_hints, *gem.type_hints]))[:12]
    date_snippets = list(dict.fromkeys([*base.date_snippets, *gem.date_snippets]))[:5]
    day_of_week_hints = list(dict.fromkeys([*base.day_of_week_hints, *gem.day_of_week_hints]))
    return EventAttributes(
        event_id=base.event_id,
        price_free_signal=base.price_free_signal,
        price_paid_signal=base.price_paid_signal,
        price_amount_rsd=base.price_amount_rsd,
        price_amount_eur=base.price_amount_eur,
        city_hint=city,
        date_snippets=date_snippets,
        day_of_week_hints=day_of_week_hints,
        type_hints=type_hints,
        noise_level_hint=noise,
        outdoor_hint=outdoor,
        extraction_method="rules_v2_gemini_lazy",
    )


def extract_event_attributes_maybe_lazy_gemini(event: NormalizedEvent) -> EventAttributes:
    base = extract_event_attributes(event)
    if not _lazy_gemini_env_enabled() or base.noise_level_hint != "unknown":
        return base
    try:
        from belgrade_recommender.llm.gemini_event_attributes import parse_event_attributes_with_gemini
        gem = parse_event_attributes_with_gemini(event)
    except (ImportError, RuntimeError, ValueError, OSError):
        return base
    return _merge_rules_with_gemini_lazy(base, gem)
