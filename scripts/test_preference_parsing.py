"""
Run: python test_preference_parsing.py
Output saved to: test_preference_parsing_output.txt
"""

import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from belgrade_recommender.llm.openai_preferences import parse_preferences_openai

QUERIES = [
    # noise + party
    "organizujem devojacko vece, hocemo da plesamo i pijemo cocktaile, nesto glamurozno",
    "hocu techno zurku, sto glasniju sto bolju, deep house ili techno klub",
    # quiet / conversation
    "hocu da pricam sa drustvom, nesto tiho, kafic ili mirno mesto",
    "ne podnosim buku, trazim apsolutnu tisinu, meditacija ili tiha izlozba",
    # free / budget
    "nemam para uopste, trazim nesto besplatno za vikend u Beogradu",
    "subota uvece, budzet je mali max 2000 din, hocu bar ili koncert",
    # forbidden categories
    "sve mi je ok, jedino ne volim techno, hocemo nesto gde mozemo da sedimo i pricamo",
    "mrzim techno, volim indie folk ili akuticnu muziku, nesto intimno i toplo",
    # days of week
    "petak uvece, trazim dzez bar ili dzez klub u Beogradu",
    "trazim klasicni koncert u februaru u Beogradu, besplatno bi bilo idealno",
    # food / vegan
    "ja sam veganka, trazim vegan friendly restoran ili food festival",
    "brzi radni rucak sa kolegama, imamo sat vremena, nesto brzo i ukusno u centru",
    # outdoor / nature
    "obozavam planinarenje i prirodu, trazim izlet ili turu oko Beograda za vikend",
    "hocu nesto aktivno napolju, yoga u parku, biciklizam ili vikend festival na obali",
    # multilingual Russian
    "ищу джаз-концерт или живую музыку в Белграде сегодня вечером, желательно недорого",
    "веган без глютена, не переношу табачный дым, ищу кафе с веганским меню",
    # multilingual English
    "I just moved to Belgrade, looking for indie or alternative rock gigs, underground",
    "looking for craft beer spots or local food events, nothing too touristy",
    # occasion-based
    "imam blind date, zelim nesto opusteno gde mozemo da razgovaramo, nije previse glasno",
    "organizujem team izlazak za medjunarodnu ekipu, hocu gde ce se svi osecati prijatno, dobra hrana",

    # family / kids
    "trazimo nesto za decu, porodicno, bezbedno i zabavno, ne preglasno",
    "organizujem izlazak za komsiluk, ima dece, odraslih i penzionera, trazim nesto na otvorenom besplatno",
    "nedeljni porodicni rucak, dolaze i roditelji i deca, srpska kuhinja, toplo i opusteno",

    # celebration / occasions
    "zavrsio sam sve ispite! slavimo sa drugarima, budzet je mali ali hocu dobru zurku ili koncert",
    "organizujem iznenadjenje za rodjendan za drugaricu Jelenu, ona obozava jazz muziku i dobru hranu",
    "momacko vece, hocu dobar bar ili pub gde mozemo da se smestimo u grupi, dobro pivo i neka zabava",

    # food specific
    "obozavam azijsku kuhinju, sushi, ramen, trazim dobar azijski restoran ili food event",
    "volim domace srpsko jelo, rostilj, nesto tradicionalno i autenticno",
    "hocu all-you-can-eat meso, rostilj, pljeskavice, janjetina, dobar srpski grill",
    "ja sam raw vegan, jedem iskljucivo sirovu neobradjenu biljnu hranu, trazim raw vegan restoran",

    # neighborhoods
    "hocu da se prosetam uz Kalemegdan ili Stari Grad, nesto kulturno napolju, lepe panorame",
    "hocu u Savamali, volim kreativne prostore, galerije, popup izlozbe i alternativnu scenu",
    "zivim u Novom Beogradu, hocu nesto blizu Usca, sport, piknik ili neka manifestacija napolju",

    # underground / alternative scene
    "trazim gothic industrial ili dark ambient koncert ili klub, mracna estetika, crno oblacenje",
    "trazim mesta koja se ne pojavljuju u turistickim vodicima, lokalna skrivena mesta, vinyl muzika",
    "trazim speakeasy barove, skriveni barovi, ulaz kroz telefonsku govornicu, visoki koktel craft mixology",
    "trazim bushcraft ili wilderness survival radionice oko Beograda, kampovanje bez opreme",

    # tech / professional
    "trazim tech meetup ili startup dogadjaj u Beogradu, networking je bitan",
    "interesuju me radionice o vestavtackoj inteligenciji i novim tehnologijama",
    "organizujem izlazak za medjunarodne kolege posle konferencije, dobar restoran pa pice",

    # live music specific
    "volim dzez, trazim dzez koncert ili bar sa dzez muzikom u Beogradu",
    "preferam klasicnu muziku ili operu, nesto ozbiljno i kvalitetno",
    "hocu da cuje srpsku narodnu muziku, kafana sa zivom svirkom, starogradske pesme",
    "trazim akusticni nastup, singer-songwriter ili akusticni bend, mali prostor, intimna atmosfera",
    "gitarista sam, trazim open mic ili jam session u Beogradu gde mogu da sviram",

    # board games / niche
    "trazim board game kafic u Beogradu koji ima Catan, Ticket to Ride, Pandemic i slicne igre",
    "trazim kafic u Beogradu koji je poznat po tome da ljudi rade, uce ili citaju, tiho, ni biblioteka ni klub",

    # Russian multilingual
    "хочу попробовать аутентичную сербскую кухню, хороший ресторан с национальными блюдами",
    "интересуют выставки и галереи современного искусства, можно и бесплатно",
    "хочу что-нибудь живое — концерт или вечеринка, желательно не слишком громко",
    "хочу провести ночь за философскими разговорами в баре с книгами, тихая атмосфера, открыто до поздна",
    "ищу языковой обмен или социальные мероприятия где русские и сербы общаются",

    # English multilingual
    "I want to experience authentic Belgrade culture, local music or art, something unique",
    "interested in galleries, design events, or photography exhibitions in Belgrade",
    "just arrived in Belgrade, first evening, want authentic local food and drinks, not touristy",
    "I moved to Belgrade 2 months ago and don't know many people, looking for social events or expat meetups",
    "looking for a tabletop RPG venue or board game cafe in Belgrade where I can play D&D",
    "looking for a jam session or open mic night in Belgrade where musicians get together, I play drums",

    # very specific constraints
    "hocu nesto aktivno ali ne skupo, max 1500 din, napolju, subota popodne",
    "trazim nesto za sredu uvece od 20h, dzez ili ziva muzika, nisam za techno",
    "ne pijem alkohol i ne podnosim dim, trazim smoke-free prostor sa bezalkoholnim opcijama",
    "dolazim sa ogromnim psom, treba mi pet friendly kafic ili basta gde mogu da udjem sa velikim psom",
    "imam alergiju na orasaste plodove i celijakiju, trazim restoran bez glutena i bez orasastih",

    # late night / time specific
    "zavrsio sam smenu u 5 ujutru, hocu nesto tiho gde mogu da sedim uz caj i opustim se",
    "hocemo da pijemo i razgovaramo o filozofiji do 4 ujutru, kafic ili bar koji radi kasno",
    "vstajem u 5 ujutru pre treninga, trazim proteinski dorucak ili ovsenu kasu, zdravo, brzo",

    # outdoor / sport / nature
    "hocu rooftop bar ili lepu terasu u Beogradu, hocu pogled na grad i dobra pica",
    "trazim letnji festival u avgustu u Beogradu, muzika i hrana napolju, dan i noc",
    "nedeljno popodne, hocu nesto opusteno, kafic sa karakterom ili manja izlozba",

    # art / creative
    "studentkinja sam likovne umetnosti, trazim izlozbe, otvorene atelije ili street art, besplatno je plus",
    "fotografkinja, trazim izlozbe fotografije, otvorene studije, ili mesta koja fotografima izgledaju interesantno",
    "vodim stranog prijatelja kroz Beograd, hocu da mu pokazem sve sto volim, dobra kafa, kulturna mesta, pa vecer sa muzikom",
]


def fmt(result) -> str:
    hc = result.hard_constraints
    sp = result.soft_preferences
    parts = []
    if sp.liked_types:            parts.append(f"liked        {sp.liked_types}")
    if sp.noise_tolerance != "unknown": parts.append(f"noise        {sp.noise_tolerance}")
    if hc.must_be_free:           parts.append( "free         True")
    if hc.budget_max_rsd:         parts.append(f"budget_rsd   {hc.budget_max_rsd}")
    if hc.forbidden_event_categories: parts.append(f"forbidden    {hc.forbidden_event_categories}")
    if hc.venue_requirements:     parts.append(f"venue_req    {hc.venue_requirements}")
    if hc.dietary_restrictions:   parts.append(f"dietary      {hc.dietary_restrictions}")
    if hc.preferred_days_of_week: parts.append(f"days         {hc.preferred_days_of_week}")
    if hc.earliest_start_hour_weekday is not None:
                                  parts.append(f"from_hour    {hc.earliest_start_hour_weekday}:00")
    if sp.preferred_time_windows: parts.append(f"time_windows {sp.preferred_time_windows}")
    if hc.city:                   parts.append(f"city         {hc.city}")
    return "\n    ".join(parts) if parts else "(nothing extracted)"


total = len(QUERIES)
ok = 0
lines = [f"Preference parsing test — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"]

def out(text=""):
    print(text)
    lines.append(text)

for i, query in enumerate(QUERIES, 1):
    label = f"[{i:02d}/{total}]"
    short = query[:80] + ("..." if len(query) > 80 else "")
    out(f"\n{'-'*65}")
    out(f"{label} {short}")
    out(f"{'-'*65}")
    try:
        r = parse_preferences_openai(query)
        out(f"  summary: {r.summary_one_line or '-'}")
        out(f"    {fmt(r)}")
        ok += 1
    except Exception as e:
        out(f"  ERROR: {e}")

out(f"\n{'='*65}")
out(f"  {ok}/{total} queries parsed successfully")
out("="*65)

output_path = Path(__file__).parent / "test_preference_parsing_output.txt"
output_path.write_text("\n".join(lines), encoding="utf-8")
print(f"\nOutput saved to: {output_path}")
