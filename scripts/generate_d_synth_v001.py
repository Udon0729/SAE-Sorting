"""Generate D_synth_v001 synthetic knowledge graph (dataset.md §3).

v001 scope (intentional): C1-C4 (relational categories only).
  C5 synthetic_biology, C6 math_formula, C7 synthetic_rule deferred to v001.5.

Categories and relations:
  C1 person_attribute: born_in (person -> city)
  C2 geography:        located_in (city -> country)
  C3 organization:     headquartered_in (org -> city)
  C4 occupation:       works_as (person -> occupation)

Cross-category relation reuse for Phase 3 entanglement:
  located_in is reused for org -> city, spanning C2 and C3.

4-phase generation (dataset.md §3.5.6):
  Phase 1: category-separated baseline facts
  Phase 2: entity-level entanglement (person also gets works_as, spanning C1+C4)
  Phase 3: relation-level entanglement (located_in extended to org -> city, spanning C2+C3)
  Phase 4: compositional chains (person -> org -> city -> country)

Annotation count targets per dataset.md §3.5.6:
  small:  entity 200 / relation 50 / compositional 30
  main:   entity 800 / relation 200 / compositional 100
  full:   entity 2000 / relation 500 / compositional 250

Outputs (dataset.md §3.4): data/synthetic_kg/v001/{config.yaml, seed.txt,
  entities/*.jsonl, relations.jsonl, facts.jsonl, entanglement_annotations.jsonl,
  templates.yaml, corpus/{train,valid,test}.txt, qa/*.jsonl, utility/*.jsonl,
  hashes/sha256sums.txt}.

Run:
  uv run python scripts/generate_d_synth_v001.py --size small
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_BASE = REPO_ROOT / "data" / "synthetic_kg" / "v001"
CONFIGS = REPO_ROOT / "configs"

SEED = 42
VERSION = "v001"

# -- size knobs ----------------------------------------------------------------

SIZE_PROFILES = {
    "small": {
        "phase1_per_relation": {"born_in": 2500, "located_in": 3000, "headquartered_in": 2500, "works_as": 2000},
        "n_entity_annotations": 200,
        "n_relation_annotations": 50,
        "n_compositional_chains": 30,
        "n_utility_prompts": 200,
    },
    "main": {
        "phase1_per_relation": {"born_in": 12500, "located_in": 15000, "headquartered_in": 12500, "works_as": 10000},
        "n_entity_annotations": 800,
        "n_relation_annotations": 200,
        "n_compositional_chains": 100,
        "n_utility_prompts": 1000,
    },
    "full": {
        "phase1_per_relation": {"born_in": 25000, "located_in": 30000, "headquartered_in": 25000, "works_as": 20000},
        "n_entity_annotations": 2000,
        "n_relation_annotations": 500,
        "n_compositional_chains": 250,
        "n_utility_prompts": 2000,
    },
}

# -- entity vocab pools (procedural, avoids pretrained-knowledge contamination) ---

PERSON_FIRST = [
    "Lior", "Aren", "Cael", "Mira", "Solen", "Tarn", "Yvel", "Brex", "Doran", "Elen",
    "Faren", "Goth", "Halen", "Iren", "Jorel", "Kael", "Lant", "Maren", "Noren", "Oren",
    "Pira", "Quen", "Ronel", "Soral", "Talen", "Uren", "Vera", "Wren", "Xanel", "Yren",
    "Zoral", "Aben", "Bral", "Cyren", "Drel", "Eral", "Foren", "Glen", "Hylen", "Iren",
    "Jaral", "Kyrel", "Loran", "Mylen", "Nyrel", "Oryn", "Pylen", "Quoral", "Riven", "Syral",
    "Tylen", "Uren", "Velen", "Wylen", "Xyren", "Yvel", "Zylen", "Aelen", "Boran", "Cyren",
    "Dralen", "Eylen", "Fyren", "Gralen", "Hyren", "Iralen", "Jylen", "Kyren", "Lyralen", "Mryen",
    "Nyralen", "Olen", "Pralen", "Qyren", "Ryalen", "Syren", "Tylen", "Vyralen", "Wyren", "Xyralen",
    "Yralen", "Zyren", "Aralen", "Beralen", "Cylen", "Dyren", "Eyralen", "Fylen", "Gyralen", "Hylen",
    "Iyren", "Jyralen", "Kylen", "Lyren", "Myralen", "Nylen", "Oyralen", "Pylen", "Qyralen", "Rylen",
]
PERSON_LAST = [
    "Vane", "Korr", "Estrin", "Halt", "Brex", "Voss", "Caldrin", "Veron", "Tylor", "Jeran",
    "Westra", "Mortel", "Daren", "Breth", "Solen", "Wynth", "Aldren", "Frix", "Goven", "Halor",
    "Ivren", "Jordal", "Kelron", "Lithen", "Marex", "Norven", "Othren", "Pravel", "Quaron", "Rolen",
    "Sythen", "Trovel", "Uthen", "Volent", "Westen", "Xeran", "Ywen", "Zoren", "Avren", "Brilen",
    "Cythen", "Drelen", "Estren", "Frylen", "Gervel", "Halren", "Ivornel", "Jorlen", "Krelen", "Lyren",
    "Mythen", "Norel", "Othrelen", "Praven", "Quolen", "Rolven", "Synthel", "Trolven", "Uvel", "Volenor",
    "Wystrel", "Xelven", "Ylvor", "Zolven", "Aprelen", "Bryven", "Cylvor", "Dyvelor", "Eythrel", "Fyralen",
    "Gyvol", "Hyralen", "Iyvor", "Jyralen", "Kyvelor", "Lyralen", "Myvelor", "Nyralen", "Oyvolt", "Pyralen",
    "Qyvor", "Rylven", "Sylvor", "Tylven", "Uyvor", "Vyrlen", "Wynvor", "Xylven", "Yvolen", "Zyrlen",
    "Aldoven", "Brivlen", "Cyrlen", "Dryven", "Estren", "Frylen", "Gyrven", "Hylven", "Iyrlen", "Jyrven",
]
CITY_PREFIX = [
    "Cal", "Vel", "Ner", "Bre", "Sol", "Tor", "Wyn", "Pal", "Mor", "Hal",
    "Ger", "Lor", "Tyr", "Fre", "Bel", "Rou", "Var", "Ken", "Mar", "Os",
    "Ank", "Doz", "Ely", "Far", "Gru", "Hyk", "Ilv", "Jor", "Kry", "Lov",
    "Mev", "Noz", "Olv", "Pyk", "Qor", "Ryv", "Syk", "Tov", "Uvy", "Vrek",
    "Wol", "Xyr", "Yov", "Zur", "Aly", "Bol", "Cev", "Dol", "Ery", "Fov",
]
CITY_SUFFIX = [
    "drin", "loria", "ovia", "thal", "wyck", "burn", "shire", "ford", "haven", "mark",
    "stead", "wold", "ridge", "bridge", "port", "quay", "moor", "fell", "gate", "view",
    "vale", "glen", "mere", "garth", "field", "hold", "rest", "hollow", "spire", "reach",
    "borough", "minster", "mouth", "town", "ham", "stoke", "leigh", "thwaite", "pool", "wick",
    "by", "side", "cross", "well", "scar", "scape", "barrow", "down", "dean", "bourn",
]
COUNTRY_PREFIX = [
    "Ar", "Ber", "Cyl", "Dyl", "Eyr", "Fyr", "Gyl", "Hyr", "Ivr", "Jyl",
    "Kyr", "Lyr", "Myr", "Nyl", "Oyr", "Pyl", "Qyr", "Ryl", "Syl", "Tyl",
    "Uyr", "Vyl", "Wyr", "Xyl", "Yyr", "Zyl", "Aer", "Bel", "Cer", "Der",
]
COUNTRY_SUFFIX = [
    "anos", "esia", "iria", "olia", "uria", "ynia", "alia", "esia", "ira", "oria",
    "umia", "ymia", "anos", "anor", "enia", "oria", "ulia", "yria", "amia", "enor",
]
ORG_PREFIX = [
    "Arven", "Brexil", "Caldon", "Drestor", "Eltrin", "Fyrov", "Glaron", "Halven", "Ivrol", "Jaron",
    "Krelven", "Lithor", "Marvol", "Nyrov", "Olrith", "Pylven", "Qaron", "Rylvor", "Sythel", "Tylven",
    "Uvron", "Velron", "Wystol", "Xylor", "Yorel", "Zylven", "Avenor", "Bryvor", "Cyrven", "Dralven",
    "Eyrol", "Fryven", "Gyron", "Hylven", "Iyron", "Jyrven", "Kylvor", "Lyrven", "Myrol", "Nyrven",
    "Olrven", "Pyron", "Qyrven", "Ryron", "Syrven", "Tyrol", "Uyron", "Vyrven", "Wyron", "Xyrven",
]
ORG_TYPE = [
    "Institute", "Foundation", "Society", "Council", "Guild", "Consortium",
    "Bureau", "Trust", "Association", "Collective", "Syndicate", "Authority",
    "Order", "Academy", "Conservatory", "Lyceum", "Sodality", "Chamber",
]
OCCUPATION_LIST = [
    "archivist", "cartographer", "lapidary", "chronologer", "lexicographer", "philologist",
    "topographer", "ethnographer", "campanologist", "horologist", "vexillologist", "calligrapher",
    "epigrapher", "phonologist", "xylographer", "tinsmith", "cooper", "wheelwright",
    "fletcher", "wainwright", "cordwainer", "currier", "hosier", "saddler",
    "thatcher", "millwright", "gleaner", "verger", "sextant_maker", "navigator",
    "ostler", "ferrier", "drover", "bargeman", "lighthouse_keeper", "harpsichordist",
    "viol_maker", "luthier", "bookbinder", "scrivener", "illuminator", "rubricator",
    "armorer", "bowyer", "alchemist", "glassblower", "coppersmith", "joiner",
    "cabinetmaker", "carpenter", "stonemason", "plasterer", "tiler", "brickmaker",
    "potter", "tanner", "fuller", "weaver", "dyer", "embroiderer",
    "tailor", "milliner", "hatter", "glover", "furrier", "bonnet_maker",
    "cordwainer_apprentice", "barber_surgeon", "apothecary", "midwife", "herbalist", "limner",
    "miniaturist", "fresco_painter", "muralist", "tapestry_weaver", "tablet_carver", "wax_chandler",
    "candle_maker", "soapboiler", "rope_maker", "sail_maker", "net_maker", "fisher",
    "trapper", "fowler", "huntsman", "venator", "warden", "forester",
    "vintner", "brewer", "miller", "baker", "confectioner", "chandler",
    "draper", "haberdasher", "pewterer", "goldsmith",
]


# -- helpers -------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gen_unique(rng: random.Random, gen_fn, target: int) -> list[str]:
    """Generate a sorted unique list of `target` strings via gen_fn(rng)."""
    seen: set[str] = set()
    attempts = 0
    cap = max(target * 50, 1000)
    while len(seen) < target and attempts < cap:
        seen.add(gen_fn(rng))
        attempts += 1
    if len(seen) < target:
        raise RuntimeError(f"Could not generate {target} unique entities (got {len(seen)})")
    return sorted(seen)


def gen_person(rng: random.Random) -> str:
    return f"{rng.choice(PERSON_FIRST)} {rng.choice(PERSON_LAST)}"


def gen_city(rng: random.Random) -> str:
    return rng.choice(CITY_PREFIX) + rng.choice(CITY_SUFFIX)


def gen_country(rng: random.Random) -> str:
    return rng.choice(COUNTRY_PREFIX) + rng.choice(COUNTRY_SUFFIX)


def gen_org(rng: random.Random) -> str:
    return f"{rng.choice(ORG_PREFIX)} {rng.choice(ORG_TYPE)}"


# -- relations and templates ---------------------------------------------------

RELATIONS = [
    {
        "relation_id": "born_in",
        "category": "person_attribute",
        "subj_type": "person",
        "obj_type": "city",
        "templates_train": [
            "{s} was born in {o}.",
            "The birthplace of {s} is {o}.",
            "{s} comes from {o}.",
        ],
        "templates_test": [
            "{s} originates from {o}.",
            "In the registry, {s}'s birth city is listed as {o}.",
        ],
    },
    {
        "relation_id": "located_in",
        "category": "geography",
        "subj_type": "city",
        "obj_type": "country",
        "templates_train": [
            "{s} is located in {o}.",
            "{s} lies within {o}.",
            "The city of {s} belongs to {o}.",
        ],
        "templates_test": [
            "{s} is part of {o}.",
            "{s}, a city in {o}, is well known.",
        ],
    },
    {
        "relation_id": "headquartered_in",
        "category": "organization",
        "subj_type": "organization",
        "obj_type": "city",
        "templates_train": [
            "{s} is headquartered in {o}.",
            "The headquarters of {s} are in {o}.",
            "{s} has its main office in {o}.",
        ],
        "templates_test": [
            "{s} operates from {o}.",
            "{o} hosts the headquarters of {s}.",
        ],
    },
    {
        "relation_id": "works_as",
        "category": "occupation",
        "subj_type": "person",
        "obj_type": "occupation",
        "templates_train": [
            "{s} works as {a_o}.",
            "{s}'s profession is {a_o}.",
            "{s} earns a living as {a_o}.",
        ],
        "templates_test": [
            "By trade, {s} is {a_o}.",
            "{s} practices the trade of {o}.",
        ],
    },
    # Phase 3 relation-level entanglement: located_in reused for org -> city
    {
        "relation_id": "org_located_in",
        "category": "organization",
        "subj_type": "organization",
        "obj_type": "city",
        "templates_train": [
            "{s} is located in {o}.",
            "{s} lies within {o}.",
            "The organization {s} belongs to {o}.",
        ],
        "templates_test": [
            "{s} is part of {o}.",
            "{s}, an organization in {o}, is well established.",
        ],
        "shares_surface_with": "located_in",
    },
]


def article(word: str) -> str:
    return ("an " if word and word[0].lower() in "aeiou" else "a ") + word


def render(template: str, s: str, o: str) -> str:
    a_o = article(o.replace("_", " "))
    o_disp = o.replace("_", " ")
    return template.format(s=s, o=o_disp, a_o=a_o)


# -- generation phases ---------------------------------------------------------

def gen_entities(rng: random.Random, profile: dict) -> dict[str, list[str]]:
    # Each entity should appear in ~10-30 facts on average for clean profiling.
    # Constraint is n_subj * n_obj >= n_facts (pair-uniqueness), not n_subj >= n_facts.
    n_persons = max(profile["phase1_per_relation"]["born_in"] // 20, 200)
    n_cities = max(profile["phase1_per_relation"]["located_in"] // 20, 200)
    n_countries = max(profile["phase1_per_relation"]["located_in"] // 30, 50)
    n_orgs = max(profile["phase1_per_relation"]["headquartered_in"] // 20, 200)
    return {
        "person": gen_unique(rng, gen_person, n_persons),
        "city": gen_unique(rng, gen_city, n_cities),
        "country": gen_unique(rng, gen_country, n_countries),
        "organization": gen_unique(rng, gen_org, n_orgs),
        "occupation": sorted(set(OCCUPATION_LIST)),
    }


def phase1_facts(rng: random.Random, entities: dict, profile: dict) -> list[dict]:
    """Category-separated baseline facts (one fact per relation per draw, no entanglement)."""
    facts: list[dict] = []
    for relation in RELATIONS:
        if relation["relation_id"] == "org_located_in":
            continue  # injected only in Phase 3
        n = profile["phase1_per_relation"][relation["relation_id"]]
        subj_type = relation["subj_type"]
        obj_type = relation["obj_type"]
        # use sampling without replacement on (subj, obj) pairs to avoid duplicates
        seen: set[tuple[str, str]] = set()
        cap = n * 20
        attempts = 0
        while len(seen) < n and attempts < cap:
            s = rng.choice(entities[subj_type])
            o = rng.choice(entities[obj_type])
            seen.add((s, o))
            attempts += 1
        if len(seen) < n:
            raise RuntimeError(f"phase1: could not generate {n} unique pairs for {relation['relation_id']} (got {len(seen)})")
        for s, o in sorted(seen):
            facts.append({
                "fact_id": f"f_{len(facts):08d}",
                "s": s, "r": relation["relation_id"], "o": o,
                "category": relation["category"],
                "phase": 1,
            })
    return facts


def phase2_entity_entanglement(
    rng: random.Random,
    facts: list[dict],
    entities: dict,
    profile: dict,
) -> tuple[list[dict], list[dict]]:
    """Pick persons; add a works_as fact each. Each adds an entity-level annotation."""
    n = profile["n_entity_annotations"]
    persons_in_p1 = sorted({f["s"] for f in facts if f["r"] == "born_in"})
    chosen = rng.sample(persons_in_p1, min(n, len(persons_in_p1)))

    new_facts: list[dict] = []
    annotations: list[dict] = []
    for p in chosen:
        o = rng.choice(entities["occupation"])
        fact = {
            "fact_id": f"f_p2_{len(new_facts):06d}",
            "s": p, "r": "works_as", "o": o,
            "category": "occupation",
            "phase": 2,
        }
        new_facts.append(fact)
        annotations.append({
            "annotation_id": f"ent_v001_{len(annotations):06d}",
            "entanglement_type": "entity",
            "categories_involved": ["person_attribute", "occupation"],
            "entities_involved": [f"person:{p}"],
            "relations_involved": ["born_in", "works_as"],
            "prompt_ids": [],  # populated after QA rendering
            "description": f"{p} appears in person_attribute (born_in) and occupation (works_as)",
        })
    return new_facts, annotations


def phase3_relation_entanglement(
    rng: random.Random,
    entities: dict,
    profile: dict,
    annotation_offset: int,
) -> tuple[list[dict], list[dict]]:
    """org_located_in facts (org -> city). Same surface 'located_in' relation as C2 (city -> country).
    Each new fact gets a relation-level annotation tying C2 and C3 via shared 'located_in' surface."""
    n = profile["n_relation_annotations"]
    new_facts: list[dict] = []
    annotations: list[dict] = []
    seen: set[tuple[str, str]] = set()
    while len(new_facts) < n:
        s = rng.choice(entities["organization"])
        o = rng.choice(entities["city"])
        if (s, o) in seen:
            continue
        seen.add((s, o))
        fact = {
            "fact_id": f"f_p3_{len(new_facts):06d}",
            "s": s, "r": "org_located_in", "o": o,
            "category": "organization",
            "phase": 3,
        }
        new_facts.append(fact)
        annotations.append({
            "annotation_id": f"ent_v001_{annotation_offset + len(annotations):06d}",
            "entanglement_type": "relation",
            "categories_involved": ["geography", "organization"],
            "entities_involved": [],
            "relations_involved": ["located_in", "org_located_in"],
            "prompt_ids": [],
            "description": "located_in surface appears in geography (city->country) and organization (org->city)",
        })
    return new_facts, annotations


def phase4_compositional(
    rng: random.Random,
    entities: dict,
    profile: dict,
    annotation_offset: int,
) -> tuple[list[dict], list[dict]]:
    """Build chains person -> works_at -> org -> headquartered_in -> city -> located_in -> country.
    Each chain produces 3 facts and 1 compositional annotation."""
    n = profile["n_compositional_chains"]
    new_facts: list[dict] = []
    annotations: list[dict] = []
    used_persons: set[str] = set()
    while len(annotations) < n:
        p = rng.choice(entities["person"])
        if p in used_persons:
            continue
        used_persons.add(p)
        org = rng.choice(entities["organization"])
        city = rng.choice(entities["city"])
        country = rng.choice(entities["country"])
        chain_facts = [
            {"fact_id": f"f_p4_{len(new_facts):06d}", "s": p, "r": "works_as", "o": rng.choice(entities["occupation"]), "category": "occupation", "phase": 4},
            {"fact_id": f"f_p4_{len(new_facts)+1:06d}", "s": org, "r": "headquartered_in", "o": city, "category": "organization", "phase": 4},
            {"fact_id": f"f_p4_{len(new_facts)+2:06d}", "s": city, "r": "located_in", "o": country, "category": "geography", "phase": 4},
        ]
        new_facts.extend(chain_facts)
        annotations.append({
            "annotation_id": f"ent_v001_{annotation_offset + len(annotations):06d}",
            "entanglement_type": "compositional",
            "categories_involved": ["occupation", "organization", "geography"],
            "entities_involved": [f"person:{p}", f"org:{org}", f"city:{city}", f"country:{country}"],
            "relations_involved": ["works_as", "headquartered_in", "located_in"],
            "prompt_ids": [],
            "description": f"chain: {p} -> {org} -> {city} -> {country}",
        })
    return new_facts, annotations


# -- renderers (corpus, qa, utility) ------------------------------------------

def render_corpus(facts: list[dict], rng: random.Random) -> tuple[list[str], list[str], list[str]]:
    """Render each fact with a randomly-chosen TRAIN template; partition 80/10/10."""
    rel_map = {r["relation_id"]: r for r in RELATIONS}
    sentences = []
    for f in facts:
        rel = rel_map[f["r"]]
        tmpl = rng.choice(rel["templates_train"])
        sentences.append(render(tmpl, f["s"], f["o"]))
    rng.shuffle(sentences)
    n = len(sentences)
    n_train = int(n * 0.8)
    n_valid = int(n * 0.1)
    return sentences[:n_train], sentences[n_train:n_train+n_valid], sentences[n_train+n_valid:]


def render_qa(facts: list[dict], rng: random.Random) -> dict[str, list[dict]]:
    """Build QA splits per dataset.md §3.4.

    Splits:
      train_qa            : 80% of facts, train templates
      valid_qa            : 10% of facts, train templates
      test_seen           : remaining 10% of facts, train templates (held-out facts, seen templates)
      test_unseen_template: subset of test_seen, regenerated with TEST templates
      test_heldout_entity : facts whose subject was held out from train
      test_compositional  : QA over phase 4 chains (3-hop reasoning)
      test_contrast       : per-fact contrast pair (real obj vs random other obj of same type)
    """
    rel_map = {r["relation_id"]: r for r in RELATIONS}
    facts_shuffled = list(facts)
    rng.shuffle(facts_shuffled)

    # 5% of subjects held out from train (heldout_entity test)
    all_subjects = sorted({f["s"] for f in facts})
    rng.shuffle(all_subjects)
    n_heldout = max(int(len(all_subjects) * 0.05), 10)
    heldout_subjects = set(all_subjects[:n_heldout])

    train_facts, valid_facts, test_seen_facts = [], [], []
    heldout_facts = []
    for f in facts_shuffled:
        if f["s"] in heldout_subjects:
            heldout_facts.append(f)
        else:
            # 80/10/10 of non-heldout
            r = rng.random()
            if r < 0.80:
                train_facts.append(f)
            elif r < 0.90:
                valid_facts.append(f)
            else:
                test_seen_facts.append(f)

    def make_qa(f, template_pool: str = "train") -> dict:
        rel = rel_map[f["r"]]
        tmpls = rel["templates_" + template_pool]
        tmpl = rng.choice(tmpls)
        # Mask the object: turn fact into a question-completion.
        # Strategy: render the full sentence, then strip the object and trailing punctuation.
        full = render(tmpl, f["s"], f["o"])
        o_disp = f["o"].replace("_", " ")
        a_o = article(o_disp)
        for variant in (a_o, o_disp):
            idx = full.rfind(variant)
            if idx != -1:
                prompt = full[:idx].rstrip()
                # strip trailing connectives that imply the answer is right after
                return {
                    "qa_id": f"qa_{f['fact_id']}_{template_pool}",
                    "fact_id": f["fact_id"],
                    "category": f["category"],
                    "relation": f["r"],
                    "subject": f["s"],
                    "answer": o_disp,
                    "prompt": prompt,
                    "template_pool": template_pool,
                }
        return {
            "qa_id": f"qa_{f['fact_id']}_{template_pool}",
            "fact_id": f["fact_id"],
            "category": f["category"],
            "relation": f["r"],
            "subject": f["s"],
            "answer": o_disp,
            "prompt": full,
            "template_pool": template_pool,
        }

    train_qa = [make_qa(f, "train") for f in train_facts]
    valid_qa = [make_qa(f, "train") for f in valid_facts]
    test_seen = [make_qa(f, "train") for f in test_seen_facts]
    test_unseen_template = [make_qa(f, "test") for f in test_seen_facts]
    test_heldout_entity = [make_qa(f, "train") for f in heldout_facts]

    # compositional: phase 4 chains - QA over the country given the person via 3 hops
    p4_facts = [f for f in facts if f["phase"] == 4]
    p4_by_chain: dict[str, list[dict]] = {}
    for f in p4_facts:
        # phase 4 facts come in groups of 3 with sequential fact_id
        cid = f["fact_id"][:8]  # group by f_p4_NNNNNN truncated
        # Better: group by phase4 chain index (3 facts per chain in insertion order)
    test_compositional = []
    for i in range(0, len(p4_facts), 3):
        if i + 2 >= len(p4_facts):
            break
        f_works, f_hq, f_loc = p4_facts[i], p4_facts[i+1], p4_facts[i+2]
        # Multi-hop QA: given person, ask country
        prompt = (
            f"{f_works['s']} works as {f_works['o'].replace('_',' ')}. "
            f"{f_hq['s']} is headquartered in {f_hq['o']}. "
            f"{f_loc['s']} is located in"
        )
        test_compositional.append({
            "qa_id": f"qa_comp_{i//3:06d}",
            "category": "compositional",
            "relation_chain": ["works_as", "headquartered_in", "located_in"],
            "answer": f_loc["o"],
            "prompt": prompt,
            "fact_ids": [f_works["fact_id"], f_hq["fact_id"], f_loc["fact_id"]],
        })

    # contrast: for each test_seen QA, build a pair with a wrong answer of the same type
    test_contrast = []
    by_obj_type: dict[str, list[str]] = {}
    for r in RELATIONS:
        for f in facts:
            if f["r"] == r["relation_id"]:
                by_obj_type.setdefault(r["obj_type"], []).append(f["o"])
    for r in RELATIONS:
        seen_set = set(by_obj_type.get(r["obj_type"], []))
        by_obj_type[r["obj_type"]] = sorted(seen_set)
    for q in test_seen[:min(len(test_seen), 2000)]:
        rel = rel_map[q["relation"]]
        candidates = by_obj_type.get(rel["obj_type"], [])
        if len(candidates) < 2:
            continue
        wrong = q["answer"]
        attempts = 0
        while wrong == q["answer"] and attempts < 10:
            wrong = rng.choice(candidates)
            attempts += 1
        if wrong == q["answer"]:
            continue
        test_contrast.append({
            "qa_id": f"qa_contrast_{q['fact_id']}",
            "category": q["category"],
            "relation": q["relation"],
            "subject": q["subject"],
            "true_answer": q["answer"],
            "false_answer": wrong,
            "prompt": q["prompt"],
        })

    return {
        "train_qa": train_qa,
        "valid_qa": valid_qa,
        "test_seen": test_seen,
        "test_unseen_template": test_unseen_template,
        "test_heldout_entity": test_heldout_entity,
        "test_compositional": test_compositional,
        "test_contrast": test_contrast,
    }


def render_utility(rng: random.Random, n: int) -> list[dict]:
    """Simple utility prompts: arithmetic, alphabetical ordering, repetition. Category-agnostic."""
    out = []
    op_choices = ["+", "-", "*"]
    for i in range(n // 4):
        a, b = rng.randint(1, 50), rng.randint(1, 50)
        op = rng.choice(op_choices)
        ans = {"+": a + b, "-": a - b, "*": a * b}[op]
        out.append({"qa_id": f"util_arith_{i:04d}", "type": "arithmetic", "prompt": f"{a} {op} {b} =", "answer": str(ans)})
    letters = "abcdefghijklmnopqrstuvwxyz"
    for i in range(n // 4):
        idx = rng.randint(0, len(letters) - 2)
        out.append({"qa_id": f"util_alpha_{i:04d}", "type": "alphabetical", "prompt": f"The letter that comes after {letters[idx]} is", "answer": letters[idx + 1]})
    for i in range(n // 4):
        word = "".join(rng.choices("abcdefghij", k=4))
        out.append({"qa_id": f"util_repeat_{i:04d}", "type": "repetition", "prompt": f"Repeat the word: {word}. Answer:", "answer": word})
    while len(out) < n:
        a, b = rng.randint(1, 100), rng.randint(1, 100)
        out.append({"qa_id": f"util_pad_{len(out):04d}", "type": "arithmetic", "prompt": f"{a} + {b} =", "answer": str(a + b)})
    return out[:n]


# -- I/O helpers ---------------------------------------------------------------

def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for l in lines:
            f.write(l + "\n")


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


# -- main ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=list(SIZE_PROFILES.keys()), default="small")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    profile = SIZE_PROFILES[args.size]
    rng = random.Random(args.seed)

    out = OUT_BASE  # all sizes share the v001 directory; size identified by config
    out.mkdir(parents=True, exist_ok=True)

    print(f"--- D_synth_v001 / size={args.size} / seed={args.seed} ---")

    print("[entities]")
    entities = gen_entities(rng, profile)
    plural = {"person": "persons", "city": "cities", "country": "countries", "organization": "organizations", "occupation": "occupations"}
    for k, v in entities.items():
        print(f"  {k}: {len(v)}")
        write_jsonl(out / "entities" / f"{plural[k]}.jsonl", [{"name": x, "type": k} for x in v])

    print("[relations]")
    write_jsonl(out / "relations.jsonl", [
        {k: v for k, v in r.items() if k != "templates_train" and k != "templates_test"}
        for r in RELATIONS
    ])

    print("[templates]")
    write_yaml(out / "templates.yaml", {
        r["relation_id"]: {
            "train": r["templates_train"],
            "test": r["templates_test"],
        } for r in RELATIONS
    })

    print("[phase 1] category-separated facts")
    facts = phase1_facts(rng, entities, profile)
    print(f"  facts: {len(facts)}")

    print("[phase 2] entity-level entanglement")
    p2_facts, p2_anns = phase2_entity_entanglement(rng, facts, entities, profile)
    facts.extend(p2_facts)
    print(f"  facts +{len(p2_facts)}, annotations +{len(p2_anns)}")

    print("[phase 3] relation-level entanglement")
    p3_facts, p3_anns = phase3_relation_entanglement(rng, entities, profile, annotation_offset=len(p2_anns))
    facts.extend(p3_facts)
    print(f"  facts +{len(p3_facts)}, annotations +{len(p3_anns)}")

    print("[phase 4] compositional chains")
    p4_facts, p4_anns = phase4_compositional(rng, entities, profile, annotation_offset=len(p2_anns) + len(p3_anns))
    facts.extend(p4_facts)
    print(f"  facts +{len(p4_facts)}, annotations +{len(p4_anns)}")

    annotations = p2_anns + p3_anns + p4_anns
    print(f"[totals] facts={len(facts)} annotations={len(annotations)}")

    write_jsonl(out / "facts.jsonl", facts)
    write_jsonl(out / "entanglement_annotations.jsonl", annotations)

    print("[corpus] rendering")
    train_text, valid_text, test_text = render_corpus(facts, rng)
    write_lines(out / "corpus" / "train.txt", train_text)
    write_lines(out / "corpus" / "valid.txt", valid_text)
    write_lines(out / "corpus" / "test.txt", test_text)
    print(f"  train={len(train_text)} valid={len(valid_text)} test={len(test_text)}")

    print("[qa] rendering")
    qa = render_qa(facts, rng)
    for split, rows in qa.items():
        write_jsonl(out / "qa" / f"{split}.jsonl", rows)
        print(f"  {split}: {len(rows)}")

    print("[utility] rendering")
    util = render_utility(rng, profile["n_utility_prompts"])
    write_jsonl(out / "utility" / "utility_prompts.jsonl", util)
    print(f"  utility: {len(util)}")

    write_yaml(out / "config.yaml", {
        "dataset_name": "D_synth_v001",
        "version": VERSION,
        "size": args.size,
        "seed": args.seed,
        "frozen_at": now_iso(),
        "categories": ["person_attribute", "geography", "organization", "occupation"],
        "categories_deferred": ["synthetic_biology", "math_formula", "synthetic_rule"],
        "n_facts": len(facts),
        "n_annotations_by_type": {
            "entity": len(p2_anns),
            "relation": len(p3_anns),
            "compositional": len(p4_anns),
        },
        "entity_counts": {k: len(v) for k, v in entities.items()},
    })
    (out / "seed.txt").write_text(f"{args.seed}\n")

    # hashes/sha256sums.txt for all output files (excluding hashes/ itself)
    print("[hashes] computing sha256sums")
    files = sorted(p for p in out.rglob("*") if p.is_file() and "hashes" not in p.parts)
    sums_lines = []
    for p in files:
        sums_lines.append(f"{sha256_of(p)}  {p.relative_to(out)}")
    write_lines(out / "hashes" / "sha256sums.txt", sums_lines)
    sums_path = out / "hashes" / "sha256sums.txt"
    sums_sha = sha256_of(sums_path)

    # update configs/hash_log.json
    hash_log_path = CONFIGS / "hash_log.json"
    hash_log = json.loads(hash_log_path.read_text()) if hash_log_path.exists() else {}
    hash_log[f"d_synth_v001_{args.size}_sums_hash"] = {
        "file": str(sums_path.relative_to(REPO_ROOT)),
        "sha256": sums_sha,
        "n_facts": len(facts),
        "n_annotations": len(annotations),
        "frozen_at": now_iso(),
    }
    hash_log_path.write_text(json.dumps(hash_log, indent=2, ensure_ascii=False, sort_keys=True))
    print(f"\nsha256sums.txt: {sums_path}  sha256={sums_sha}")
    print(f"hash_log updated: {hash_log_path}")


if __name__ == "__main__":
    main()
