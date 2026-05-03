"""Generate D_utility_v001 (dataset.md §6).

Categories (U1-U6):
  U1 general_lm_text      : wikitext-103-raw-v1 validation, prefix+continuation
  U2 simple_qa            : procedural common-sense QA templates
  U3 syntax_format        : procedural format-following prompts
  U4 numeric              : procedural arithmetic
  U5 copying_list         : procedural repetition / list continuation
  U6 unrelated_factual    : hand-curated science/math facts (sampled with replacement)

Sizes (per dataset.md §6):
  small : ~500 prompts  (Pythia / Qwen pilot)
  main  : ~2000 prompts (Pythia main / Qwen main)
  full  : ~10000 prompts (controlled LM / robustness)

Outputs (data/utility/v001/):
  config.yaml, seed.txt
  prompts.jsonl                       (all U1-U6 unified)
  categories/U{1..6}.jsonl            (per-category split)
  hashes/sha256sums.txt
Plus snapshot: data/raw/wikitext/wikitext_103_raw_v1_validation.parquet

Run:
  uv run python scripts/generate_d_utility_v001.py --size small
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import yaml
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_BASE = REPO_ROOT / "data" / "utility" / "v001"
DATA_RAW = REPO_ROOT / "data" / "raw"
CONFIGS = REPO_ROOT / "configs"

SEED = 42
VERSION = "v001"
WIKITEXT_DATASET = "wikitext"
WIKITEXT_CONFIG = "wikitext-103-raw-v1"
WIKITEXT_SPLIT = "validation"

SIZE_PROFILES = {
    "small": {"per_category": 90},  # 6 * 90 = 540
    "main":  {"per_category": 340}, # 6 * 340 = 2040
    "full":  {"per_category": 1700},# 6 * 1700 = 10200
}

CATEGORY_DEFS = [
    ("U1", "general_lm_text"),
    ("U2", "simple_qa"),
    ("U3", "syntax_format"),
    ("U4", "numeric"),
    ("U5", "copying_list_continuation"),
    ("U6", "unrelated_factual"),
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


# -- U1: general LM text from wikitext snapshot --------------------------------

def freeze_wikitext_snapshot() -> tuple[Path, str]:
    snap = DATA_RAW / "wikitext" / f"{WIKITEXT_CONFIG.replace('-', '_')}_{WIKITEXT_SPLIT}.parquet"
    cache = DATA_RAW / "wikitext_cache"
    if snap.exists():
        return snap, sha256_of(snap)
    snap.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(WIKITEXT_DATASET, WIKITEXT_CONFIG, split=WIKITEXT_SPLIT, cache_dir=str(cache))
    df = ds.to_pandas()
    df.to_parquet(snap, index=False)
    return snap, sha256_of(snap)


def gen_u1(rng: random.Random, n: int, snap: Path) -> list[dict]:
    import pandas as pd
    df = pd.read_parquet(snap)
    paragraphs = sorted(set(t for t in df["text"].tolist() if isinstance(t, str)))
    # filter for usable length (skip headings, very short or very long lines)
    usable = [p.strip() for p in paragraphs if 200 <= len(p.strip()) <= 1000 and not p.strip().startswith("=")]
    if len(usable) < n:
        raise RuntimeError(f"U1: only {len(usable)} usable paragraphs in {snap.name}, need {n}")
    rng.shuffle(usable)
    out: list[dict] = []
    for i, p in enumerate(usable):
        if len(out) >= n:
            break
        words = p.split()
        if len(words) < 30:
            continue
        prefix_len = rng.randint(15, max(16, len(words) - 11))
        prefix = " ".join(words[:prefix_len])
        cont = " ".join(words[prefix_len:prefix_len + 10])
        out.append({
            "prompt_id": f"u1_{len(out):06d}",
            "category_id": "U1",
            "category_name": "general_lm_text",
            "type_tag": "completion",
            "prompt": prefix,
            "expected_continuation": cont,
            "metadata": {
                "source": f"{WIKITEXT_DATASET}/{WIKITEXT_CONFIG}@{WIKITEXT_SPLIT}",
                "n_prefix_words": prefix_len,
            },
        })
    return out


# -- U2: simple QA -------------------------------------------------------------

U2_TEMPLATES = [
    ("What color is the {thing}?", "color"),
    ("How many {part}s does a {thing} have?", "count"),
    ("Where do {animal}s typically live?", "habitat"),
    ("What sound does a {animal} make?", "sound"),
    ("What is the opposite of {word}?", "antonym"),
]
U2_VOCAB = {
    "color": [("sky", "blue"), ("grass", "green"), ("snow", "white"), ("blood", "red"),
              ("sun", "yellow"), ("coal", "black"), ("milk", "white"), ("lemon", "yellow")],
    "count": [("hand", "finger", "5"), ("octopus", "leg", "8"), ("week", "day", "7"),
              ("year", "month", "12"), ("triangle", "side", "3"), ("square", "side", "4")],
    "habitat": [("fish", "in water"), ("bird", "in trees"), ("bear", "in forests"),
                ("camel", "in deserts"), ("whale", "in the ocean"), ("monkey", "in jungles")],
    "sound": [("dog", "bark"), ("cat", "meow"), ("cow", "moo"), ("sheep", "baa"),
              ("duck", "quack"), ("lion", "roar"), ("horse", "neigh"), ("frog", "croak")],
    "antonym": [("hot", "cold"), ("big", "small"), ("up", "down"), ("fast", "slow"),
                ("light", "dark"), ("happy", "sad"), ("rich", "poor"), ("near", "far")],
}


def gen_u2(rng: random.Random, n: int) -> list[dict]:
    out: list[dict] = []
    while len(out) < n:
        for template, kind in U2_TEMPLATES:
            if len(out) >= n:
                break
            if kind == "color":
                thing, ans = rng.choice(U2_VOCAB["color"])
                prompt = template.format(thing=thing)
            elif kind == "count":
                thing, part, ans = rng.choice(U2_VOCAB["count"])
                prompt = template.format(thing=thing, part=part)
            elif kind == "habitat":
                animal, ans = rng.choice(U2_VOCAB["habitat"])
                prompt = template.format(animal=animal)
            elif kind == "sound":
                animal, ans = rng.choice(U2_VOCAB["sound"])
                prompt = template.format(animal=animal)
            elif kind == "antonym":
                word, ans = rng.choice(U2_VOCAB["antonym"])
                prompt = template.format(word=word)
            out.append({
                "prompt_id": f"u2_{len(out):06d}",
                "category_id": "U2",
                "category_name": "simple_qa",
                "type_tag": "qa",
                "prompt": prompt,
                "expected_continuation": ans,
                "metadata": {"template_kind": kind},
            })
    return out[:n]


# -- U3: syntax / format -------------------------------------------------------

U3_TEMPLATES = [
    ("Output as JSON: a key '{key}' with value {value}.", "json"),
    ("List 3 {thing}s separated by commas:", "list"),
    ("Convert to upper case: '{word}'.", "case_upper"),
    ("Convert to lower case: '{word}'.", "case_lower"),
    ("Wrap '{word}' in parentheses:", "wrap"),
    ("Reverse the word '{word}':", "reverse"),
]
U3_KEYS = ["name", "age", "color", "city", "score", "level", "size", "kind", "rank", "year"]
U3_VALUES = ["42", "100", "0", "1", "true", "false", "null", "7", "12", "99"]
U3_THINGS = ["fruit", "color", "animal", "country", "vehicle", "tool", "instrument", "metal", "shape", "season"]
U3_WORDS = ["hello", "world", "computer", "language", "python", "model", "tensor", "feature", "cluster", "vector"]


def gen_u3(rng: random.Random, n: int) -> list[dict]:
    out: list[dict] = []
    while len(out) < n:
        for template, kind in U3_TEMPLATES:
            if len(out) >= n:
                break
            if kind == "json":
                k = rng.choice(U3_KEYS); v = rng.choice(U3_VALUES)
                prompt = template.format(key=k, value=v)
                ans = f'{{"{k}": {v}}}'
            elif kind == "list":
                thing = rng.choice(U3_THINGS)
                prompt = template.format(thing=thing)
                ans = "<3 items>"
            elif kind == "case_upper":
                w = rng.choice(U3_WORDS); prompt = template.format(word=w); ans = w.upper()
            elif kind == "case_lower":
                w = rng.choice(U3_WORDS); prompt = template.format(word=w.upper()); ans = w
            elif kind == "wrap":
                w = rng.choice(U3_WORDS); prompt = template.format(word=w); ans = f"({w})"
            elif kind == "reverse":
                w = rng.choice(U3_WORDS); prompt = template.format(word=w); ans = w[::-1]
            out.append({
                "prompt_id": f"u3_{len(out):06d}",
                "category_id": "U3",
                "category_name": "syntax_format",
                "type_tag": "format_following",
                "prompt": prompt,
                "expected_continuation": ans,
                "metadata": {"template_kind": kind},
            })
    return out[:n]


# -- U4: numeric ---------------------------------------------------------------

def gen_u4(rng: random.Random, n: int) -> list[dict]:
    out: list[dict] = []
    while len(out) < n:
        op = rng.choice(["+", "-", "*"])
        a = rng.randint(1, 99)
        b = rng.randint(1, 99)
        ans = {"+": a + b, "-": a - b, "*": a * b}[op]
        out.append({
            "prompt_id": f"u4_{len(out):06d}",
            "category_id": "U4",
            "category_name": "numeric",
            "type_tag": "arithmetic",
            "prompt": f"{a} {op} {b} =",
            "expected_continuation": str(ans),
            "metadata": {"op": op, "a": a, "b": b},
        })
    return out


# -- U5: copying / list continuation -------------------------------------------

def gen_u5(rng: random.Random, n: int) -> list[dict]:
    out: list[dict] = []
    letters = "abcdefghijklmnopqrstuvwxyz"
    digits = "0123456789"
    while len(out) < n:
        kind = rng.choice(["repeat_word", "letter_seq", "number_seq", "ascending"])
        if kind == "repeat_word":
            word = "".join(rng.choices("abcdefghij", k=rng.randint(3, 5)))
            prompt = f"Repeat: {word} {word} {word}. Next:"
            ans = word
        elif kind == "letter_seq":
            start = rng.randint(0, len(letters) - 5)
            seq = letters[start:start + 4]
            prompt = f"Continue the sequence: {seq[0]} {seq[1]} {seq[2]} ___"
            ans = seq[3]
        elif kind == "number_seq":
            start = rng.randint(1, 90)
            prompt = f"Continue: {start}, {start+1}, {start+2}, ___"
            ans = str(start + 3)
        else:  # ascending
            base = rng.randint(2, 8)
            prompt = f"Continue: {base}, {base*2}, {base*3}, ___"
            ans = str(base * 4)
        out.append({
            "prompt_id": f"u5_{len(out):06d}",
            "category_id": "U5",
            "category_name": "copying_list_continuation",
            "type_tag": "sequence",
            "prompt": prompt,
            "expected_continuation": ans,
            "metadata": {"kind": kind},
        })
    return out


# -- U6: unrelated factual (curated, no overlap with C1-C4) --------------------

U6_FACTS = [
    ("Water freezes at", "0 degrees Celsius"),
    ("Water boils at", "100 degrees Celsius"),
    ("The chemical formula for water is", "H2O"),
    ("The Earth orbits the", "Sun"),
    ("The Sun is a", "star"),
    ("A triangle has", "3 sides"),
    ("A square has", "4 sides"),
    ("A pentagon has", "5 sides"),
    ("A hexagon has", "6 sides"),
    ("Pi is approximately", "3.14"),
    ("A year has", "12 months"),
    ("A week has", "7 days"),
    ("A day has", "24 hours"),
    ("An hour has", "60 minutes"),
    ("A minute has", "60 seconds"),
    ("Light travels faster than", "sound"),
    ("Sound travels in", "waves"),
    ("Magnets have two", "poles"),
    ("The opposite of north is", "south"),
    ("The opposite of east is", "west"),
    ("The first prime number is", "2"),
    ("The smallest positive integer is", "1"),
    ("The number of legs on an insect is", "6"),
    ("The number of legs on a spider is", "8"),
    ("The metal liquid at room temperature is", "mercury"),
    ("The hardest natural substance is", "diamond"),
    ("Plants need sunlight for", "photosynthesis"),
    ("The atmosphere is mostly composed of", "nitrogen"),
    ("Oxygen makes up about 21 percent of the", "atmosphere"),
    ("The largest planet in the Solar System is", "Jupiter"),
    ("The closest star to Earth other than the Sun is", "Proxima Centauri"),
    ("The Moon orbits the", "Earth"),
    ("Tides are caused by the gravity of the", "Moon"),
    ("Sound cannot travel in a", "vacuum"),
    ("The boiling point of water at sea level is", "100 degrees Celsius"),
    ("Helium is lighter than", "air"),
    ("A circle has degrees totaling", "360"),
    ("A right angle measures", "90 degrees"),
    ("The opposite of multiplication is", "division"),
    ("The opposite of addition is", "subtraction"),
    ("The square root of 16 is", "4"),
    ("The square root of 25 is", "5"),
    ("The square root of 100 is", "10"),
    ("Two times two equals", "four"),
    ("Three times three equals", "nine"),
    ("Five times five equals", "twenty-five"),
    ("Ten times ten equals", "one hundred"),
    ("The Roman numeral for one is", "I"),
    ("The Roman numeral for five is", "V"),
    ("The Roman numeral for ten is", "X"),
    ("The Roman numeral for fifty is", "L"),
    ("The Roman numeral for one hundred is", "C"),
    ("The Roman numeral for one thousand is", "M"),
    ("Sound is measured in", "decibels"),
    ("Distance is measured in", "meters"),
    ("Mass is measured in", "kilograms"),
    ("Time is measured in", "seconds"),
    ("Temperature is measured in", "degrees"),
    ("Force is measured in", "newtons"),
    ("Pressure is measured in", "pascals"),
    ("Energy is measured in", "joules"),
    ("Power is measured in", "watts"),
    ("Frequency is measured in", "hertz"),
    ("The boiling point of nitrogen is below", "zero"),
    ("Iron is attracted to", "magnets"),
    ("Glass is made primarily from", "sand"),
    ("Bread is made from", "flour"),
    ("Honey is made by", "bees"),
    ("Silk is produced by", "silkworms"),
    ("Wool comes from", "sheep"),
    ("Leather comes from animal", "skin"),
    ("Coal is a fossil", "fuel"),
    ("Oil is a fossil", "fuel"),
    ("Solar panels convert sunlight into", "electricity"),
    ("Wind turbines convert wind into", "electricity"),
    ("Hydroelectric dams convert flowing water into", "electricity"),
    ("The opposite of inhale is", "exhale"),
    ("The lung is part of the body's", "respiratory system"),
    ("The heart is part of the body's", "circulatory system"),
    ("The brain is part of the body's", "nervous system"),
    ("The stomach is part of the body's", "digestive system"),
    ("The kidney is part of the body's", "urinary system"),
    ("Three sides of equal length make a triangle that is", "equilateral"),
    ("A four-sided polygon is called a", "quadrilateral"),
    ("A many-sided polygon is generally called a", "polygon"),
    ("The smallest unit of matter that retains chemical identity is the", "atom"),
    ("Two or more atoms bonded together form a", "molecule"),
    ("The center of an atom is called the", "nucleus"),
    ("Particles that orbit the nucleus are called", "electrons"),
    ("The chemical symbol for gold is", "Au"),
    ("The chemical symbol for silver is", "Ag"),
    ("The chemical symbol for iron is", "Fe"),
    ("The chemical symbol for oxygen is", "O"),
    ("The chemical symbol for nitrogen is", "N"),
    ("The chemical symbol for carbon is", "C"),
    ("The chemical symbol for hydrogen is", "H"),
    ("Ice is the solid form of", "water"),
    ("Steam is the gaseous form of", "water"),
    ("Liquid is the form between solid and", "gas"),
    ("Sound waves require a", "medium"),
]


def gen_u6(rng: random.Random, n: int) -> list[dict]:
    out: list[dict] = []
    indices = list(range(len(U6_FACTS)))
    while len(out) < n:
        rng.shuffle(indices)
        for i in indices:
            if len(out) >= n:
                break
            prompt, ans = U6_FACTS[i]
            out.append({
                "prompt_id": f"u6_{len(out):06d}",
                "category_id": "U6",
                "category_name": "unrelated_factual",
                "type_tag": "factual_completion",
                "prompt": prompt,
                "expected_continuation": ans,
                "metadata": {"source_index": i},
            })
    return out


# -- main ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=list(SIZE_PROFILES.keys()), default="small")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    profile = SIZE_PROFILES[args.size]
    n_per = profile["per_category"]
    rng = random.Random(args.seed)

    out = OUT_BASE
    out.mkdir(parents=True, exist_ok=True)

    print(f"--- D_utility_v001 / size={args.size} / seed={args.seed} ---")

    print("[U1] freezing wikitext snapshot")
    snap, snap_sha = freeze_wikitext_snapshot()
    print(f"  snapshot: {snap.relative_to(REPO_ROOT)}  sha256={snap_sha[:16]}...")

    generators = {
        "U1": lambda: gen_u1(rng, n_per, snap),
        "U2": lambda: gen_u2(rng, n_per),
        "U3": lambda: gen_u3(rng, n_per),
        "U4": lambda: gen_u4(rng, n_per),
        "U5": lambda: gen_u5(rng, n_per),
        "U6": lambda: gen_u6(rng, n_per),
    }
    all_prompts: list[dict] = []
    for cat_id, cat_name in CATEGORY_DEFS:
        prompts = generators[cat_id]()
        print(f"  {cat_id} {cat_name}: {len(prompts)}")
        write_jsonl(out / "categories" / f"{cat_id}.jsonl", prompts)
        all_prompts.extend(prompts)

    write_jsonl(out / "prompts.jsonl", all_prompts)
    write_yaml(out / "config.yaml", {
        "dataset_name": "D_utility_v001",
        "version": VERSION,
        "size": args.size,
        "seed": args.seed,
        "frozen_at": now_iso(),
        "categories": [{"id": cid, "name": cname} for cid, cname in CATEGORY_DEFS],
        "n_per_category": n_per,
        "n_total": len(all_prompts),
        "u1_source": {
            "dataset": WIKITEXT_DATASET,
            "config": WIKITEXT_CONFIG,
            "split": WIKITEXT_SPLIT,
            "snapshot": str(snap.relative_to(REPO_ROOT)),
            "snapshot_sha256": snap_sha,
        },
    })
    (out / "seed.txt").write_text(f"{args.seed}\n")

    # sha256sums.txt for everything except hashes/
    print("[hashes] computing sha256sums")
    files = sorted(p for p in out.rglob("*") if p.is_file() and "hashes" not in p.parts)
    sums_lines = [f"{sha256_of(p)}  {p.relative_to(out)}" for p in files]
    write_lines(out / "hashes" / "sha256sums.txt", sums_lines)
    sums_path = out / "hashes" / "sha256sums.txt"
    sums_sha = sha256_of(sums_path)

    hash_log_path = CONFIGS / "hash_log.json"
    hash_log = json.loads(hash_log_path.read_text()) if hash_log_path.exists() else {}
    hash_log["wikitext_validation_snapshot_hash"] = {
        "file": str(snap.relative_to(REPO_ROOT)),
        "sha256": snap_sha,
        "source": f"{WIKITEXT_DATASET}/{WIKITEXT_CONFIG}@{WIKITEXT_SPLIT}",
        "frozen_at": now_iso(),
    }
    hash_log[f"d_utility_v001_{args.size}_sums_hash"] = {
        "file": str(sums_path.relative_to(REPO_ROOT)),
        "sha256": sums_sha,
        "n_total": len(all_prompts),
        "n_per_category": n_per,
        "frozen_at": now_iso(),
    }
    hash_log_path.write_text(json.dumps(hash_log, indent=2, ensure_ascii=False, sort_keys=True))

    print(f"\nsha256sums.txt: {sums_path}  sha256={sums_sha}")
    print(f"hash_log updated: {hash_log_path}")
    print(f"total prompts: {len(all_prompts)}")


if __name__ == "__main__":
    main()
