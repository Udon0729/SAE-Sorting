"""Generate D_probe_v001 from D_fact_v001 sources (dataset.md §5).

Build summary:
  - Sources: LAMA T-REx + PopQA + CounterFact (D_synth excluded; controlled-LM only)
  - Categories: C1-C4 only (matches D_synth_v001 v001 scope)
  - Sizes: small (300/cat), main (1000/cat), full (5000+/cat) via --size
  - Prompt types in v001: positive + paraphrase + control (3 of 6 types)
      * Deferred to v002: same-subject contrast, same-object-type contrast,
        same-template contrast (last requires §5.3 GPT-4o pre-check)
  - Splits: random fit/validate/generalize 60/20/20 with contrast_group_id integrity
      * v002 TODO: relation-held-out generalize per §5.2

Per-category target distribution (small):
  positive    : 180  (60%)
  paraphrase  :  60  (20%, linked to ~30 positive parents via contrast_group_id)
  control     :  60  (20%, sampled from D_utility_v001 U6)
  total       : 300

Schema follows dataset.md §9 (qid, source, category, category_type, relation,
subject, object, prompt, answer, prompt_type, contrast_group_id, template_id,
split, split_role, popularity_score, popularity_bucket).

Outputs:
  data/probe/v001/{config.yaml, prompts.jsonl, by_category/C{1..4}.jsonl,
                   by_split/{fit,validate,generalize}.jsonl, hashes/sha256sums.txt}

Run:
  uv run python scripts/generate_d_probe_v001.py --size small
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_BASE = REPO_ROOT / "data" / "probe" / "v001"
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_UTILITY = REPO_ROOT / "data" / "utility" / "v001"
CONFIGS = REPO_ROOT / "configs"
MAPPING = CONFIGS / "probe" / "relation_to_category.yaml"
POPULARITY_DEF = CONFIGS / "popularity_bucket_definition.yaml"

SEED = 42
VERSION = "v001"

CAT_KEYS = ["person_attribute", "geography", "organization", "occupation"]
CAT_ID = {"person_attribute": "C1", "geography": "C2", "organization": "C3", "occupation": "C4"}

SIZE_PROFILES = {
    "small": {"per_category": 300, "n_positive": 180, "n_paraphrase": 60, "n_control": 60},
    "main":  {"per_category": 1000, "n_positive": 600, "n_paraphrase": 200, "n_control": 200},
    "full":  {"per_category": 5000, "n_positive": 3000, "n_paraphrase": 1000, "n_control": 1000},
}


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


def assign_popularity_bucket(score: float, low_max: float, mid_max: float) -> str:
    if score <= low_max:
        return "low"
    if score <= mid_max:
        return "mid"
    return "high"


# -- ingestion -----------------------------------------------------------------

def load_mapping() -> dict:
    return yaml.safe_load(MAPPING.read_text())


def load_popularity_def() -> tuple[float, float]:
    pop = yaml.safe_load(POPULARITY_DEF.read_text())
    return float(pop["boundaries"]["low_max"]), float(pop["boundaries"]["mid_max"])


def load_lama_facts(mapping: dict) -> list[dict]:
    df = pd.read_parquet(DATA_RAW / "lama" / "lama_trex.parquet")
    lama_map = mapping["lama_trex"]
    facts = []
    for _, row in df.iterrows():
        rel = row.get("predicate_id")
        if rel not in lama_map:
            continue
        spec = lama_map[rel]
        s = row.get("sub_label")
        o = row.get("obj_label")
        if not isinstance(s, str) or not isinstance(o, str) or not s or not o:
            continue
        facts.append({
            "source": "LAMA",
            "relation": rel,
            "subject": s,
            "object": o,
            "category": spec["category"],
            "template": spec["template"],
        })
    return facts


def load_popqa_facts(mapping: dict, low_max: float, mid_max: float) -> list[dict]:
    df = pd.read_parquet(DATA_RAW / "popqa" / "popqa_test.parquet")
    popqa_map = mapping["popqa"]
    facts = []
    for _, row in df.iterrows():
        prop = row.get("prop")
        if prop not in popqa_map:
            continue
        s = row.get("subj")
        o = row.get("obj")
        question = row.get("question")
        if not isinstance(s, str) or not isinstance(o, str) or not isinstance(question, str):
            continue
        s_pop = float(row.get("s_pop", 0)) if row.get("s_pop") else 0.0
        score = math.log10(s_pop) if s_pop > 0 else None
        bucket = assign_popularity_bucket(score, low_max, mid_max) if score is not None else "n/a"
        # PopQA prompt is the full question; answer expected after.
        facts.append({
            "source": "PopQA",
            "relation": prop,
            "subject": s,
            "object": o,
            "category": popqa_map[prop]["category"],
            "prompt": question,  # use directly
            "popularity_score": score,
            "popularity_bucket": bucket,
        })
    return facts


def load_counterfact_facts(mapping: dict) -> list[dict]:
    df = pd.read_parquet(DATA_RAW / "counterfact" / "counterfact.parquet")
    cf_map = mapping["counterfact_relations"]
    facts = []
    for _, row in df.iterrows():
        rel = row.get("requested_rewrite.relation_id")
        if rel not in cf_map:
            continue
        s = row.get("requested_rewrite.subject") if "requested_rewrite.subject" in df.columns else None
        if s is None:
            # Try to extract subject from subject field if present
            for k in df.columns:
                if "subject" in k.lower():
                    s = row.get(k)
                    if isinstance(s, str):
                        break
        prompt = row.get("requested_rewrite.prompt")
        target_true = row.get("requested_rewrite.target_true.str") if "requested_rewrite.target_true.str" in df.columns else None
        paraphrase_prompts = row.get("paraphrase_prompts")
        if not isinstance(prompt, str) or not isinstance(target_true, str) or not isinstance(s, str):
            continue
        # CounterFact prompt has '{}' placeholder for subject; substitute.
        full_prompt = prompt.format(s) if "{}" in prompt else prompt
        para_list = list(paraphrase_prompts) if paraphrase_prompts is not None and hasattr(paraphrase_prompts, "__iter__") else []
        # CounterFact paraphrases also have {} -> subject substitution
        para_filled = [p.format(s) if isinstance(p, str) and "{}" in p else p for p in para_list if isinstance(p, str)]
        facts.append({
            "source": "CounterFact",
            "relation": rel,
            "subject": s,
            "object": target_true,
            "category": cf_map[rel]["category"],
            "prompt": full_prompt,
            "paraphrase_prompts": para_filled,
        })
    return facts


def load_utility_u6() -> list[dict]:
    rows = [json.loads(l) for l in (DATA_UTILITY / "categories" / "U6.jsonl").read_text().splitlines()]
    return rows


def load_pararel_patterns() -> dict[str, list[str]]:
    df = pd.read_parquet(DATA_RAW / "pararel" / "pararel_patterns.parquet")
    out: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        rel = row.get("relation_id")
        pat = row.get("pattern")
        if not isinstance(rel, str) or not isinstance(pat, str):
            continue
        out.setdefault(rel, []).append(pat)
    return out


# -- prompt assembly -----------------------------------------------------------

def render_lama_positive(fact: dict) -> str:
    return fact["template"].format(s=fact["subject"])


def render_pararel_paraphrase(rel: str, subject: str, pararel: dict[str, list[str]]) -> str | None:
    """Render a paraphrase from ParaRel pattern. Pattern uses [X] and [Y] for subject/object."""
    if rel not in pararel or not pararel[rel]:
        return None
    pat = pararel[rel][0]  # use first pattern
    if "[Y]" not in pat or "[X]" not in pat:
        return None
    # Strip "[Y]" and trailing punctuation; substitute [X] with subject.
    rendered = pat.replace("[X]", subject)
    idx = rendered.rfind("[Y]")
    prompt = rendered[:idx].rstrip(" .,").rstrip()
    return prompt if prompt else None


def make_qid(prefix: str, idx: int) -> str:
    return f"{prefix}_{idx:06d}"


def build_per_category(
    cat: str,
    cat_facts: list[dict],
    pararel: dict[str, list[str]],
    util_pool: list[dict],
    rng: random.Random,
    profile: dict,
) -> list[dict]:
    """Per-category prompt assembly: 180 positive + 60 paraphrase + 60 control."""
    n_pos = profile["n_positive"]
    n_para = profile["n_paraphrase"]
    n_ctrl = profile["n_control"]

    # Stable order of source pool: sort facts by (source, relation, subject) for determinism.
    cat_facts_sorted = sorted(cat_facts, key=lambda f: (f["source"], f["relation"], f["subject"], f.get("object", "")))
    if len(cat_facts_sorted) < n_pos:
        raise RuntimeError(f"Category {cat}: only {len(cat_facts_sorted)} facts available (need {n_pos} positive)")
    rng.shuffle(cat_facts_sorted)
    positive_facts = cat_facts_sorted[:n_pos]

    out: list[dict] = []
    cgroup = 0

    # 1. positive prompts
    for i, f in enumerate(positive_facts):
        if f["source"] == "LAMA":
            prompt = render_lama_positive(f)
            template_id = f"lama_canonical_{f['relation']}"
        else:
            prompt = f["prompt"]
            template_id = f"{f['source'].lower()}_native"
        cgid = f"{cat}_g_{cgroup:06d}"
        cgroup += 1
        row = {
            "qid": make_qid(f"probe_{CAT_ID[cat].lower()}", len(out)),
            "source": f["source"],
            "category": cat,
            "category_type": "relational",
            "relation": f["relation"],
            "subject": f["subject"],
            "object": f["object"],
            "prompt": prompt,
            "answer": f["object"],
            "prompt_type": "positive",
            "contrast_group_id": cgid,
            "template_id": template_id,
            "popularity_score": f.get("popularity_score"),
            "popularity_bucket": f.get("popularity_bucket", "n/a"),
        }
        out.append(row)

    # 2. paraphrase prompts: iterate all positive parents, take up to 2 paraphrases each,
    # stop at n_para. PopQA has no good paraphrase source so it's skipped.
    para_count = 0
    for parent_idx, parent in enumerate(positive_facts):
        if para_count >= n_para:
            break
        para_options: list[tuple[str, str]] = []
        if parent["source"] == "LAMA":
            for k, pat in enumerate(pararel.get(parent["relation"], [])[:2]):
                if "[Y]" not in pat or "[X]" not in pat:
                    continue
                rendered = pat.replace("[X]", parent["subject"])
                idx = rendered.rfind("[Y]")
                prompt = rendered[:idx].rstrip(" .,").rstrip()
                if prompt:
                    para_options.append((prompt, f"pararel_{parent['relation']}_{k}"))
        elif parent["source"] == "CounterFact":
            for k, p in enumerate(parent.get("paraphrase_prompts", [])[:2]):
                if isinstance(p, str) and p.strip():
                    para_options.append((p, f"counterfact_paraphrase_{k}"))
        cgid = out[parent_idx]["contrast_group_id"]
        for text, tid in para_options:
            if para_count >= n_para:
                break
            out.append({
                "qid": make_qid(f"probe_{CAT_ID[cat].lower()}_para", para_count),
                "source": parent["source"],
                "category": cat,
                "category_type": "relational",
                "relation": parent["relation"],
                "subject": parent["subject"],
                "object": parent["object"],
                "prompt": text,
                "answer": parent["object"],
                "prompt_type": "paraphrase",
                "contrast_group_id": cgid,
                "template_id": tid,
                "popularity_score": parent.get("popularity_score"),
                "popularity_bucket": parent.get("popularity_bucket", "n/a"),
            })
            para_count += 1

    # 3. control prompts (sample from D_utility U6, mark category context)
    util_indices = list(range(len(util_pool)))
    rng.shuffle(util_indices)
    for i in range(n_ctrl):
        ub = util_pool[util_indices[i % len(util_indices)]]
        cgid = f"{cat}_ctrl_{i:06d}"
        row = {
            "qid": make_qid(f"probe_{CAT_ID[cat].lower()}_ctrl", i),
            "source": "D_utility_v001/U6",
            "category": cat,
            "category_type": "control",
            "relation": "n/a",
            "subject": "n/a",
            "object": ub.get("expected_continuation"),
            "prompt": ub.get("prompt"),
            "answer": ub.get("expected_continuation"),
            "prompt_type": "control",
            "contrast_group_id": cgid,
            "template_id": "u6_native",
            "popularity_score": None,
            "popularity_bucket": "n/a",
        }
        out.append(row)

    return out


# -- splits --------------------------------------------------------------------

def assign_splits(prompts: list[dict], rng: random.Random) -> None:
    """Assign split_role in {fit, validate, generalize} per dataset.md §5.2.

    v001 simplification: random split with contrast_group_id integrity.
    Same contrast_group_id -> same split (paraphrase + positive parent stay together).

    Target proportions: fit=0.6, validate=0.2, generalize=0.2.
    """
    by_group: dict[str, list[int]] = {}
    for i, p in enumerate(prompts):
        by_group.setdefault(p["contrast_group_id"], []).append(i)
    group_ids = sorted(by_group.keys())
    rng.shuffle(group_ids)
    n_groups = len(group_ids)
    n_fit = int(n_groups * 0.6)
    n_val = int(n_groups * 0.2)
    fit_groups = set(group_ids[:n_fit])
    val_groups = set(group_ids[n_fit:n_fit + n_val])
    for g, idxs in by_group.items():
        if g in fit_groups:
            role = "fit"
        elif g in val_groups:
            role = "validate"
        else:
            role = "generalize"
        for i in idxs:
            prompts[i]["split"] = "probe_small"  # naming aligns with dataset.md §9 example
            prompts[i]["split_role"] = role


# -- main ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=list(SIZE_PROFILES.keys()), default="small")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    profile = SIZE_PROFILES[args.size]
    rng = random.Random(args.seed)
    out = OUT_BASE
    out.mkdir(parents=True, exist_ok=True)

    print(f"--- D_probe_v001 / size={args.size} / seed={args.seed} ---")

    print("[load] mapping + popularity_def")
    mapping = load_mapping()
    low_max, mid_max = load_popularity_def()

    print("[load] LAMA")
    lama_facts = load_lama_facts(mapping)
    print(f"  LAMA mapped facts: {len(lama_facts)}")

    print("[load] PopQA")
    popqa_facts = load_popqa_facts(mapping, low_max, mid_max)
    print(f"  PopQA mapped facts: {len(popqa_facts)}")

    print("[load] CounterFact")
    cf_facts = load_counterfact_facts(mapping)
    print(f"  CounterFact mapped facts: {len(cf_facts)}")

    print("[load] D_utility U6")
    u6_pool = load_utility_u6()
    print(f"  U6 control pool: {len(u6_pool)}")

    print("[load] ParaRel")
    pararel = load_pararel_patterns()
    print(f"  ParaRel relations available: {len(pararel)}")

    # Pool by category
    all_facts = lama_facts + popqa_facts + cf_facts
    by_cat: dict[str, list[dict]] = {c: [] for c in CAT_KEYS}
    for f in all_facts:
        if f["category"] in by_cat:
            by_cat[f["category"]].append(f)
    for c in CAT_KEYS:
        print(f"  category pool {c} ({CAT_ID[c]}): {len(by_cat[c])} facts")

    # Build prompts per category
    all_prompts: list[dict] = []
    per_cat_prompts: dict[str, list[dict]] = {}
    for c in CAT_KEYS:
        prompts = build_per_category(c, by_cat[c], pararel, u6_pool, rng, profile)
        per_cat_prompts[c] = prompts
        all_prompts.extend(prompts)
        print(f"  built {c}: {len(prompts)} prompts (positive={sum(1 for p in prompts if p['prompt_type']=='positive')}, "
              f"paraphrase={sum(1 for p in prompts if p['prompt_type']=='paraphrase')}, "
              f"control={sum(1 for p in prompts if p['prompt_type']=='control')})")

    # Splits
    print("[splits] assigning fit/validate/generalize")
    assign_splits(all_prompts, rng)
    counts = {"fit": 0, "validate": 0, "generalize": 0}
    for p in all_prompts:
        counts[p["split_role"]] += 1
    print(f"  split counts: {counts}")

    # Sort outputs deterministically by qid
    all_prompts.sort(key=lambda p: p["qid"])

    # Write outputs
    write_jsonl(out / "prompts.jsonl", all_prompts)
    for c, plist in per_cat_prompts.items():
        plist_sorted = sorted(plist, key=lambda p: p["qid"])
        write_jsonl(out / "by_category" / f"{CAT_ID[c]}.jsonl", plist_sorted)
    by_split: dict[str, list[dict]] = {"fit": [], "validate": [], "generalize": []}
    for p in all_prompts:
        by_split[p["split_role"]].append(p)
    for s, plist in by_split.items():
        write_jsonl(out / "by_split" / f"{s}.jsonl", sorted(plist, key=lambda p: p["qid"]))

    write_yaml(out / "config.yaml", {
        "dataset_name": "D_probe_v001",
        "version": VERSION,
        "size": args.size,
        "seed": args.seed,
        "frozen_at": now_iso(),
        "categories": [{"id": CAT_ID[c], "name": c} for c in CAT_KEYS],
        "categories_deferred_v002": ["synthetic_biology", "math_formula", "synthetic_rule", "behavioral"],
        "sources_used": ["LAMA", "PopQA", "CounterFact"],
        "sources_excluded_v001": ["D_synth_v001"],
        "n_per_category_target": profile["per_category"],
        "n_total": len(all_prompts),
        "prompt_types_v001": ["positive", "paraphrase", "control"],
        "prompt_types_deferred_v002": ["same_subject_contrast", "same_object_type_contrast", "same_template_contrast"],
        "split_strategy": "random_with_contrast_group_integrity",
        "split_strategy_deferred_v002": "relation_held_out_generalize_per_5.2",
        "contrast_template_validity_hash": "n/a (pending v002 §5.3 GPT-4o pre-check)",
        "split_proportions": {"fit": 0.6, "validate": 0.2, "generalize": 0.2},
        "split_counts": counts,
        "popularity_bucket_definition": str(POPULARITY_DEF.relative_to(REPO_ROOT)),
    })
    (out / "seed.txt").write_text(f"{args.seed}\n")

    # sha256sums.txt
    print("[hashes] computing sha256sums")
    files = sorted(p for p in out.rglob("*") if p.is_file() and "hashes" not in p.parts)
    sums_lines = [f"{sha256_of(p)}  {p.relative_to(out)}" for p in files]
    write_lines(out / "hashes" / "sha256sums.txt", sums_lines)
    sums_path = out / "hashes" / "sha256sums.txt"
    sums_sha = sha256_of(sums_path)

    # update hash_log
    hash_log_path = CONFIGS / "hash_log.json"
    hash_log = json.loads(hash_log_path.read_text()) if hash_log_path.exists() else {}
    hash_log[f"d_probe_v001_{args.size}_sums_hash"] = {
        "file": str(sums_path.relative_to(REPO_ROOT)),
        "sha256": sums_sha,
        "n_total": len(all_prompts),
        "split_counts": counts,
        "frozen_at": now_iso(),
    }
    hash_log_path.write_text(json.dumps(hash_log, indent=2, ensure_ascii=False, sort_keys=True))
    print(f"\nsha256sums.txt: {sums_path}  sha256={sums_sha}")
    print(f"hash_log updated: {hash_log_path}")
    print(f"total prompts: {len(all_prompts)}")


if __name__ == "__main__":
    main()
