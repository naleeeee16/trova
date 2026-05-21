"""Prompt fragments for LLM calls"""

PREFERENCE_SYSTEM_INSTRUCTION = """You extract structured preference data for a Belgrade group event recommender.
The user may write in Serbian (Latin or Cyrillic), Russian, or English.

Rules:
- Map the user's free text to the JSON schema fields only; do not invent specific venues or ticket URLs.
- Put strict requirements (free-only, budget caps, banned categories, time cutoffs) in hard_constraints.
- Put tastes and desiderata in soft_preferences.
- If the user did not mention something, leave that field null or empty list as appropriate.
- summary_one_line: ALWAYS write in English, one concise sentence, regardless of input language.

--- liked_types ---
Generate short snake_case tags describing what the user wants to experience or do.
No fixed list — infer freely. Keep tags lowercase, underscore-separated, and specific (jazz not music).
Aim for 2–5 tags. Prefer the most specific fitting tag.

SPECIFICITY RULE: When the user mentions a specific food item, drink, cuisine, or activity — keep it as-is.
Do NOT collapse it into a generic category. Generic categories ("food", "cafe") lose the signal.
Add the venue type alongside but keep the specific item:
  "hoću palačinke / pancakes / crepe" → "pancakes", "cafe"        (NOT just "food" or "cafe")
  "pizza / burger" → "pizza", "restaurant"                        (NOT just "food")
  "sushi / ramen" → "sushi", "restaurant", "asian_cuisine"        (NOT just "food")
  "sladoled / ice cream / gelato" → "ice_cream", "cafe"           (NOT just "dessert")
  "pivo / beer / craft beer" → "craft_beer", "bar"                (NOT just "bar")
  "vino / wine" → "wine", "bar"                                   (NOT just "bar")
  "kafa / coffee" → "coffee", "cafe"                              (NOT just "cafe")
  "čaj / tea" → "tea", "cafe"                                     (NOT just "cafe")
  "burek / gibanica" → "burek", "bakery"
  "doručak / breakfast items / eggs" → "breakfast", "cafe"
  Specific cocktails or spirits → keep the spirit name + "cocktail_bar" or "bar"

Format rules:
- snake_case only: live_music, craft_beer, board_game, dark_ambient
- No abstract quality words: not "safe", "fun", "glamorous", "nice", "local", "authentic" alone
- No constraint words: not "free", "cheap", "besplatno" — those go in hard_constraints
- No health/dietary words: not "smoke_free", "non_alcoholic", "gluten_free" — those go in venue_requirements / dietary_restrictions

Tag standardization (always use these exact tokens):
- Grill/barbecue: always "grill" (not rostilj, barbecue, roštilj)
- Acoustic music: always "acoustic" (not acoustic_music)
- Traditional Serbian folk/tavern music: "folk_traditional" (distinct from modern folk → "folk")
- Traditional Serbian tavern: "kafana" (not cafe — these are different venues)
- Cocktail bar: "cocktail_bar" (not cocktails, kokteli)
- Rooftop: "rooftop_bar"
- Speakeasy: "speakeasy"
- Nightclub (any genre): always include "club" alongside genre tag
- Late-night venue: "late_night_bar" (not "late", "bar_late", "kasno")
- Dance (any genre): "dance" (not "dancing", "ples" alone)
- Meditation/mindfulness: "meditation" (not "mindfulness", "relax", "zen", "mindful")
- Lounge bar: "lounge" (not "relaxed_bar", "cozy")
- Asian food: "asian_cuisine" (not "asian_food", "asian", "sushi_bar")
- Serbian traditional food: "serbian_cuisine" (not "domace", "traditional_food")
- Gallery/art space/visual art: always "exhibition" (not "gallery", "art", "galleria", "art_space")
- Gastronomy/culinary interest: "food" (not "gastronomy", "culinary", "food_culture")
- Traditional/folk/national food or music: "folk_traditional" or "serbian_cuisine" (not "traditional", "national", "domace" alone)
- Sports/physical activity: "sport" (not "sports", "fitness_activity") — always pair with "outdoor" if outdoors
- Quality/mood adjectives — NEVER use as liked_types: "romantic", "cozy", "fancy", "intimate", "underground", "hidden", "authentic", "local", "special", "unique", "relaxed", "nice". These describe a quality, not an event type. Instead, infer the actual venue or activity from the full context (e.g. "romantično vece" + dinner mention → restaurant, wine; "romantično" + "piće" → bar, lounge; "romantično" + "setnja" → outdoor; "skriveno/ne turisticko" → hidden_gem; "opusteno" → lounge or cafe).

Inference examples (Serbian / Russian / English):
- "kafic / kafe / кафе / cafe" → cafe
- "bar / pice / drinks" → bar
- "cocktail / koktel / коктейль" → cocktail_bar
- "restoran / hrana / jelo / еда / restaurant" → restaurant
- "dzez / jazz / джаз" → jazz, live_music
- "klasicna muzika / classical / классическая / opera" → classical_music, concert
- "techno / deep house / dark techno / rejv / rave" → techno, club, nightlife
- "techno klub / nightclub / klub" → techno, club, nightlife
- "splav / splavovi / na reci / river club / raft bar / brod bar" → splav, nightlife  (KEEP "splav" as-is — do NOT collapse to "club" or "river")
- "indie rock / alternative rock / gig / underground live" → concert, live_music
- "zurka / party / club / вечеринка" → party, nightlife
- "ples / dance / танцы" → dance, nightlife
- "izlozba / galerija / gallery / art / выставка / street art / art space / galleria" → exhibition
- "meditacija / meditation / тихая медитация" → meditation
- "planinarenje / hiking / izlet / природа / trekking" → outdoor, hiking
- "yoga / pilates" → yoga, outdoor, sport
- "biciklizam / cycling / trcanje / running / sport / активный / turnir / tournament / race / trka" → outdoor, sport
- "festival / vikend festival / manifestacija" → festival
- "craft beer / zanatsko pivo" → bar, craft_beer
- "vegan / veganka / веган" → restaurant (do NOT add vegan here — it goes in dietary_restrictions)
- "rostilj / roštilj / grill / meso / janjetina / pljeskavice" → grill, restaurant
- "piknik" → outdoor, food
- "board game / RPG / društvene igre / Catan / D&D" → board_game, cafe
- "open mic / jam session / svirka" → open_mic, live_music
- "team outing / team building / kolege / ekipa" → social, restaurant
- "blind date / prvi sastanak" → bar, lounge  ("romantično" alone: infer venue from context — see quality/mood adjectives rule above)
- "rodjendan / birthday / slavlje / iznenadjenje" → party, celebration
- "gothic / dark ambient / dark wave / industrial" → dark_ambient, nightlife
- "starogradska pesma / kafana / narodna muzika" → folk_traditional, kafana
- "srpska kuhinja / domace jelo / tradicionalno / national cuisine / domaća hrana" → restaurant, serbian_cuisine
- "azijska kuhinja / sushi / ramen" → restaurant, asian_cuisine
- "rooftop / terasa sa pogledom" → rooftop_bar, bar
- "speakeasy / skriveni bar" → speakeasy, bar
- "vinyl / record / ploce / vinyl market / vinilne ploce" → bar, vinyl_music
- "fotografija / photography / atelje / studio" → exhibition, photography
- "tech meetup / startup / networking / AI / vestacka inteligencija" → tech_meetup, networking
- "bushcraft / survival / kampovanje" → outdoor, bushcraft
- "expat / language exchange / jezicki" → social, language_exchange
- "anime / gaming / manga" → social, gaming
- "porodicno / deca / kids / family" → family_friendly (not "safe", "fun", "family", "children" separately)
- "skrivena mesta / hidden / lokalno / ne turisticko" → hidden_gem
- "kafica radi do kasno / bar do 4 ujutru / kasno radi" → late_night_bar
- "ujutru / dorucak / pre posla / 5 ujutru / early morning" → breakfast, early_morning
- "citanje / reading / knjige / books / literatura / book club" → lecture
- "hrana / jelo / food (generic, no specific item)" → food, restaurant  (use this ONLY when user wants food in general with no specific dish)

--- noise_tolerance ---
One of: low, medium, medium_high, high, unknown. Base on INTENT, not venue type.

- low: user wants conversation or explicit quiet — "hocu da pricam", "razgovarati", "tiho", "mirno",
  "тихо", "без музыки", "meditacija", "citaonica", "ne preglasno", "nije previse glasno",
  "не слишком громко", acoustic in intimate/small space, philosophical conversation (even in a bar),
  blind date where user says "nije previse glasno" or "opusteno razgovaramo"
- medium: casual social, restaurant dinner, team outing, family lunch, no strong music signal.
  Also: bar or cafe open late where the GOAL is conversation, not dancing
- medium_high: live music is the goal but not a loud club — jazz bar, open mic, singer-songwriter,
  folk concert, live band in restaurant
- high: explicit loud party — "zurka", "techno", "rave", "ples", "dance floor", "sto glasnije", nightclub

Key rules:
- "ne preglasno" / "nije previse glasno" / "не слишком громко" → ALWAYS low (cap hard, never medium_high or high)
- Acoustic in small/intimate space → low (not medium)
- Jazz bar for listening → medium_high
- Bar + "razgovaramo" or "da pricamo" → low
- Blind date + any quiet qualifier → low
- Bar radi do 4 ujutru / kasno radi → medium (never low — late bars are never truly quiet)
- "kafic" alone → unknown (not low)

--- forbidden_event_categories ---
ONLY set when user explicitly rejects a specific event type or genre.
Trigger phrases: "ne volim X", "mrzim X", "nisam za X", "ne podnosim [genre]", "bez X", "не люблю X".

Critical rule: If the user is SEARCHING FOR something, never put it in forbidden.
(user searching for gothic club → do NOT add "club" or "nightclub_late" to forbidden)

Critical rule: If the user describes a venue type they are NOT (e.g., "not a library, not a club"),
only add genre/category to forbidden if they explicitly dislike it — context "not X, Y instead" does
not mean "ban X".

Special rule — adults_only: Always add "adults_only" to forbidden_event_categories when:
- User mentions bringing children / says a child will attend ("imam dete", "sa detetom", "deca dolaze",
  "с ребёнком", "с детьми", "kids attending", "children coming")
- User explicitly requires family-safe / child-safe venues ("porodicno bezbedno", "sigurno za decu",
  "безопасно для детей", "family-safe", "child-friendly")
- Any context where minors are clearly present or the venue must be suitable for children.
This is a hard safety filter — adults-only events (18+, kink, fetish) must never reach family groups.

Do NOT use forbidden for:
- Dietary/health constraints → use dietary_restrictions or venue_requirements
- Noise level preferences → use noise_tolerance
- Venue types the user just didn't mention

--- venue_requirements ---
Set ONLY when user explicitly requires a venue attribute. These are strict filters, not preferences.
Allowed values: smoke_free, non_alcoholic, pet_friendly

Trigger phrases:
- "ne pijem alkohol / bez alkohola / non-alcoholic / я не пью / nisam za alkohol" → ["non_alcoholic"]
  Also add relevant venue types to liked_types: cafe, juice_bar
- "smoke-free / ne podnosim dim / без дыма / ne pusim / dim me smeta" → ["smoke_free"]
- "dolazim sa psom / pas / pet friendly / dog friendly / sa psom" → ["pet_friendly"]
  Also add relevant liked_types: cafe, outdoor

NOTE: Do NOT put these in liked_types or forbidden_event_categories.

--- dietary_restrictions ---
Set ONLY when user has an explicit dietary restriction or food allergy. These are strict requirements.
Allowed values: vegan, gluten_free, nut_free, raw_vegan, halal, lactose_free

Trigger phrases:
- "veganka / vegan / biljno / веган / растительное" → ["vegan"]; add restaurant to liked_types
- "celijakija / bez glutena / gluten free / без глютена" → ["gluten_free"]; add restaurant to liked_types
- "alergija na orasaste / nut allergy / bez orasastih / без орехов" → ["nut_free"]; add restaurant
- "raw vegan / sirova hrana / neobradjeno biljno" → ["raw_vegan"]; add restaurant
- "halal" → ["halal"]; add restaurant

NOTE: Do NOT put these in liked_types or forbidden_event_categories.
Multiple restrictions can coexist: ["gluten_free", "nut_free"]

--- must_be_free ---
Set true when the user says they want free events OR says they have no money:
- Explicit free: "besplatno", "free", "džabe", "бесплатно", "bez ulaznice", "ideally free"
- No money: "nemam para", "nema love", "bez para", "broke", "без денег", "нет денег", "нема пара"
- Combination: "nemam para, trazim besplatno" → must_be_free = true

"Jeftino" or "mali budzet" / "tight budget" → use budget_max_rsd instead, not must_be_free.

--- budget_max_rsd ---
Set for an explicit RSD cap or a clear cheap/budget statement without an exact number:
- Explicit cap: "max 2000 din" → 2000, "oko 1500" → 1500
- "jeftino / cheap / mali budzet / tight budget" with no number → 1000 (reasonable default)

--- preferred_days_of_week ---
Set ONLY when the user explicitly names a day or the exact word "weekend" / "vikend" / "выходные".
Never infer days from anything else.

Allowed triggers ONLY:
- Explicit day name: "u petak" → ["friday"], "в субботу" → ["saturday"], "on wednesday" → ["wednesday"]
- "za vikend / weekend / выходные" → ["saturday", "sunday"]

Never set days for ANY of these:
- "veceras / tonight / сегодня вечером" — time of day, not a day
- "festival / koncert / izlet / letnji festival" — event types NEVER imply days
- "ovog vikenda" alone — too vague, leave empty
- Any season, month, or event category whatsoever
- "dan i noc / day and night" — duration, not a day name

--- preferred_time_windows ---
For month hints and time-of-day context. Use plain strings.
- Month name → "february", "august", etc.
- "ujutru / breakfast / утром / dorucak" → "morning"
- "popodne / afternoon" → "afternoon"
- "uvece / evening / вечером" → "evening"
- "kasno / do 4 ujutru / radi kasno / late night / после полуночи" → "late_night"

--- earliest_start_hour_weekday ---
Set only when user wants to go out AFTER a specific hour on an outing.
"od 20h" → 20. "od 21h" → 21. "trazim nesto za sredu od 20h" → 20.

Do NOT set for:
- "vstajem u 5 ujutru pre treninga" — this is waking time, not an outing start
- "zavrsio sam smenu u 5 ujutru" — this is end of work, not an outing preference
These are context about the user's schedule, not start-hour requirements.

Respond only as schema-conformant JSON."""

RANKING_EXPLAIN_INSTRUCTION = """You explain why a set of Belgrade events was recommended to a group.

Rules:
- Write in Serbian if the preference text looks Serbian; otherwise English.
- Write 2–4 short paragraphs: one on why the top event fits, one on the group trade-offs, optionally one caveat.
- Mention the specific matching aspects: atmosphere, price, activity type, noise level.
- Do not invent facts; if price or schedule is unknown from the provided text, say so or omit.
- Do not output JSON. Plain prose only.
- Do not invent venues, links, or ticket URLs not present in the provided list.
- No condescending remarks about any person’s tastes."""

EVENT_ATTRIBUTES_LLM_INSTRUCTION = """You extract structured signals for ONE event from its hashtag list and description.
The description may be in English, Serbian, or Russian. Use only information present in the text.

Input you receive:
  tags: comma-separated hashtags (e.g. "#FreeEntry,#LiveMusic,#Exhibition")
  description: full event text (English, or English + Russian)

--- city_hint ---
belgrade | novi_sad | serbia_other | unknown

Signals → belgrade: "Belgrade", "Beograd", "Белград", or well-known Belgrade venue names
Signals → novi_sad: "Novi Sad", "Нови Сад", "Petrovaradin"
Signals → serbia_other: any named Serbian city that is not the above
If the event covers multiple cities, use the first/primary one.
unknown: no location info in the text.

--- price signals ---
price_free_signal = true when the text contains ANY of:
  "Free entry" / "Free admission" / "Вход свободный" / "Besplatno" / "Ulaz besplatan"
  "džabe" / "без улазнице" / hashtag #FreeEntry / #FreeEvent

price_paid_signal = true when there is an explicit ticket purchase link or a stated admission price.
Do NOT set price_paid_signal for incidental costs: transport, food, "lunch separately", donations.

If an event has a free opening AND paid screenings/sessions, set BOTH flags to true.

price_amount_rsd: integer from patterns like "2100 дин", "от 2000 RSD", "250 RSD", "1500 динара".
  When a range is given ("от 2000"), use the lower bound.
price_amount_eur: integer from patterns like "80 евро", "70 euros", "€80".
  When a range is given, use the lower bound.
Do NOT set these for transport, food, or non-admission costs.

--- date_snippets ---
Up to 5 literal date fragments from the text. Copy them exactly as they appear:
  "05.04.2026", "April 18", "18 апреля", "15-16 апреля", "30.03 - 05.04.2026"

--- day_of_week_hints ---
Canonical English day names ONLY when a day NAME is explicitly stated:
  "В воскресенье" / "on Sunday" / "u nedjelju" → ["sunday"]
  "В субботу" / "on Saturday" / "u subotu" → ["saturday"]
  "среда" / "Wednesday" → ["wednesday"]
Do NOT infer day from a calendar date ("April 18"). Leave empty unless the text names the day.

--- type_hints ---
Snake_case event-type tokens derived from BOTH hashtags and description text. Deduped, max 8 items.

Hashtag → type_hints mapping (use the right-hand tokens):
  #VinylMarket / #AudioEquipment → vinyl_music, market
  #MusicLovers / #LiveMusic / #Concert → live_music, concert
  #MusicFestival / #Festival → festival
  #ClassicalMusic → classical_music, concert
  #Exhibition / #Art / #DigitalArt / #VR → exhibition
  #Photography → photography, exhibition
  #FoodFestival / #Gastronomy / #Foodie / #InternationalCuisine → food_festival, restaurant
  #Networking / #BusinessForum / #Entrepreneurship → networking, lecture
  #Science / #Lecture / #PublicTalk / #Astronomy / #Education → lecture
  #Hiking / #Nature / #Outdoor → hiking, outdoor
  #GuidedTour / #Adventure → guided_tour, outdoor
  #FamilyFun / #FamilyFriendly / #EasterCarnival → family_friendly, festival
  #CraftFair → craft_fair
  #Radio / #Cafe / #Bar → cafe, bar
  #CommunityEvent / #Social → social
  #pop_punk / #live_music / #concert → live_music, concert
  #Jazz → jazz, live_music
  #Techno / #ElectronicMusic → techno, nightlife
  #Workshop / #Masterclass → workshop
  #Dance / #Salsa / #Ples / #Dancing → dance, nightlife
  #Yoga / #Pilates → yoga, outdoor, sport
  #DevFest / #TechConference / #TechEvent → tech_meetup, lecture
  #Hackathon → tech_meetup, workshop

Additional tokens from description text:
  Concert / live performance → concert
  Exhibition / gallery / выставка → exhibition
  Workshop / мастер-класс / radionica → workshop
  Lecture / лекция / predavanje → lecture
  Guided tour / экскурсия → guided_tour
  Dance / salsa / ples / tango / plesna → dance, nightlife
  Yoga / pilates / meditacija / zen yoga → yoga, outdoor, sport
  Hackathon / devfest / tech conference → tech_meetup, lecture
  Techno / DJ set / rave / club → techno, nightlife
  Jazz / Джаз / Džez → jazz
  Folk / traditional / Kafana / народная → folk_traditional
  Classical / Классика / opera → classical_music
  Festival / фестиваль → festival
  Party / zurka / вечеринка → party
  Hiking / trek / поход → hiking, outdoor
  Outdoor market / open-air → outdoor
  Restaurant / restoran / ресторан → restaurant
  Cafe / кофейня / kafana → cafe
  Bar / wine bar → bar
  Networking / startup / meetup → networking

--- noise_level_hint ---
low: lecture, exhibition, classical concert, film screening, meditation
medium: restaurant, cafe, workshop, networking, family event
medium_high: live music bar, jazz concert, open mic, singer-songwriter, folk concert in a sitting venue
high: techno/DJ event, rave, nightclub, loud outdoor festival with dancing
unknown: type is ambiguous or not enough cues

--- outdoor_hint ---
true: hiking, nature trek, outdoor market, park event, open-air festival, "napolju", "na otvorenom",
      "на улице", "на открытом воздухе", carnival in a square
false: museum, gallery room, concert hall, nightclub, indoor cafe/bar, underground catacombs
null: venue type unclear or mixed indoor/outdoor

Respond only as schema-conformant JSON."""
