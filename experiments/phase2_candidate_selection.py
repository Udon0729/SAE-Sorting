"""Phase 2 candidate selection (dataset.md §8 Phase 2).

For each (layer, feature) in phase1_features.parquet, assign one of four
candidate types using the same thresholds across models for fair comparison:

  rare              : n_firings_total < N_RARE_THRESHOLD
  category_selective: dominant cat firing_rate >= TOP1_RATE_MIN
                      AND selectivity_max_over_mean >= SELECTIVITY_MIN
  broad             : fires on >= MIN_BROAD_CATS categories each at >= BROAD_RATE_MIN
                      AND not category_selective AND not rare
  mixed             : everything else

Inputs:
  data/profiling/v001/phase1_features.parquet           (Pythia-160m, 12 layers)
  data/profiling/v001/phase1_qwen3_8b_features.parquet  (Qwen3-8B-Base, 36 layers)

Outputs:
  data/profiling/v001/phase2_candidates_pythia_160m.parquet
  data/profiling/v001/phase2_candidates_qwen3_8b.parquet
  data/profiling/v001/phase2_candidate_summary.json
  configs/hash_log.json updated with phase2_candidates_*_hash

v001 scope (intentional):
  - Per-category 4-axis selectivity only.
  - Per-relation / per-object_type axes deferred to v002 (requires joining
    raw firings with D_probe relation info).
  - Selectivity uses ALL prompt_types (positive + paraphrase + control).
    True positive-only selectivity deferred to v002 (requires raw firings
    + prompt_type filter).
  - The protocol's acceptance condition (recall(C, top 5%) >= 0.90 from
    project_candidate_validation_experiment.md) requires a controlled LM
    full causal sweep; v001 applies the protocol but does not validate it.

Contamination flags (added to support Phase 3 post-hoc stratification, NOT
used as a pre-filter):
  control_frac_top20         : fraction of top-20 activating qids with
                                prompt_type == 'control'. Universal metric
                                across all candidate types. Baseline (uniform
                                random within any cat) = 60/300 = 0.20.
  control_frac_in_dom_cat    : same as above but restricted to top-20 qids
                                that fall in dominant_category. Only meaningful
                                for category_selective candidates; NaN otherwise.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILE_DIR = REPO_ROOT / "data" / "profiling" / "v001"
CONFIGS = REPO_ROOT / "configs"
HASH_LOG = CONFIGS / "hash_log.json"
PROBE_PROMPTS = REPO_ROOT / "data" / "probe" / "v001" / "prompts.jsonl"

# Thresholds (shared across models for cross-model fair comparison)
N_RARE_THRESHOLD = 5
TOP1_RATE_MIN = 0.10
SELECTIVITY_MIN = 3.0
MIN_BROAD_CATS = 2
BROAD_RATE_MIN = 0.05

CATS = ["person_attribute", "geography", "organization", "occupation"]

MODELS = [
    {"tag": "pythia_160m", "input": "phase1_features.parquet"},
    {"tag": "qwen3_8b",    "input": "phase1_qwen3_8b_features.parquet"},
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_qid_metadata() -> tuple[dict[str, str], dict[str, str]]:
    """Build {qid: prompt_type} and {qid: category} maps from D_probe v001."""
    qid_to_type: dict[str, str] = {}
    qid_to_cat: dict[str, str] = {}
    with PROBE_PROMPTS.open("r", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            qid_to_type[p["qid"]] = p["prompt_type"]
            qid_to_cat[p["qid"]] = p["category"]
    return qid_to_type, qid_to_cat


def add_contamination_flags(
    df: pd.DataFrame,
    qid_to_type: dict[str, str],
    qid_to_cat: dict[str, str],
) -> pd.DataFrame:
    """Add control_frac_top20 (universal) and control_frac_in_dom_cat (cs-only).
    Pure additive operation: existing columns untouched."""
    def frac_top20(qids) -> float:
        if not isinstance(qids, (list, np.ndarray)) or len(qids) == 0:
            return float("nan")
        types = [qid_to_type.get(q, "unknown") for q in qids]
        return sum(1 for t in types if t == "control") / len(types)

    def frac_in_dom_cat(row) -> float:
        qids = row["top_qids"]
        dom = row["dominant_category"]
        if not isinstance(qids, (list, np.ndarray)) or len(qids) == 0 or not dom:
            return float("nan")
        in_cat = [q for q in qids if qid_to_cat.get(q) == dom]
        if not in_cat:
            return float("nan")
        return sum(1 for q in in_cat if qid_to_type.get(q) == "control") / len(in_cat)

    out = df.copy()
    out["control_frac_top20"] = out["top_qids"].apply(frac_top20)
    out["control_frac_in_dom_cat"] = out.apply(frac_in_dom_cat, axis=1)
    return out


def classify(df: pd.DataFrame) -> pd.DataFrame:
    rate_cols = [f"firing_rate_{c}" for c in CATS]
    rate_matrix = df[rate_cols].to_numpy()
    n_firings = df["n_firings_total"].to_numpy()
    selectivity = df["selectivity_max_over_mean"].to_numpy()

    top1_rate = rate_matrix.max(axis=1)
    top1_cat_idx = rate_matrix.argmax(axis=1)
    top1_cat = np.array(CATS)[top1_cat_idx]
    n_cats_above = (rate_matrix >= BROAD_RATE_MIN).sum(axis=1)

    is_rare = n_firings < N_RARE_THRESHOLD
    is_cat_sel = (top1_rate >= TOP1_RATE_MIN) & (selectivity >= SELECTIVITY_MIN) & ~is_rare
    is_broad = (n_cats_above >= MIN_BROAD_CATS) & ~is_cat_sel & ~is_rare

    candidate_type = np.full(len(df), "mixed", dtype=object)
    candidate_type[is_rare] = "rare"
    candidate_type[is_cat_sel] = "category_selective"
    candidate_type[is_broad] = "broad"

    out = df.copy()
    out["candidate_type"] = candidate_type
    out["dominant_category"] = np.where(is_cat_sel, top1_cat, "")
    out["dominant_category_rate"] = np.where(is_cat_sel, top1_rate, 0.0)
    out["n_cats_above_threshold"] = n_cats_above
    return out


def summarize(df: pd.DataFrame, model_tag: str) -> dict:
    type_counts = df["candidate_type"].value_counts().to_dict()
    type_counts = {k: int(v) for k, v in type_counts.items()}
    cat_sel_by_dom = (
        df[df["candidate_type"] == "category_selective"]
        ["dominant_category"].value_counts().to_dict()
    )
    cat_sel_by_dom = {k: int(v) for k, v in cat_sel_by_dom.items()}
    per_layer_per_type = (
        df.groupby(["layer", "candidate_type"])
          .size().unstack(fill_value=0)
          .astype(int)
    )

    # Contamination summary (no decisions made; descriptive only)
    cs = df[df["candidate_type"] == "category_selective"]
    cont_in_dom = cs["control_frac_in_dom_cat"].dropna()
    cont_top20 = df["control_frac_top20"].dropna()
    contamination = {
        "baseline_uniform_random": 60 / 300,
        "control_frac_top20_all_candidates": {
            "n_valid": int(cont_top20.size),
            "median": float(cont_top20.median()) if cont_top20.size else None,
            "mean": float(cont_top20.mean()) if cont_top20.size else None,
            "n_above_baseline": int((cont_top20 > 0.20).sum()),
            "n_above_warn": int((cont_top20 > 0.30).sum()),
            "n_above_severe": int((cont_top20 > 0.50).sum()),
        },
        "control_frac_in_dom_cat_category_selective_only": {
            "n_valid": int(cont_in_dom.size),
            "median": float(cont_in_dom.median()) if cont_in_dom.size else None,
            "mean": float(cont_in_dom.mean()) if cont_in_dom.size else None,
            "n_above_baseline": int((cont_in_dom > 0.20).sum()),
            "n_above_warn": int((cont_in_dom > 0.30).sum()),
            "n_above_severe": int((cont_in_dom > 0.50).sum()),
        },
    }

    return {
        "model": model_tag,
        "n_total_features": int(len(df)),
        "type_counts": type_counts,
        "type_counts_pct": {k: round(v / len(df) * 100, 2) for k, v in type_counts.items()},
        "category_selective_by_dominant_cat": cat_sel_by_dom,
        "per_layer_per_type": per_layer_per_type.to_dict(orient="index"),
        "n_layers": int(df["layer"].nunique()),
        "contamination": contamination,
    }


def main() -> None:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"--- Phase 2 candidate selection ---")
    print(f"  thresholds: n_rare<{N_RARE_THRESHOLD}, top1_rate>={TOP1_RATE_MIN}, "
          f"selectivity>={SELECTIVITY_MIN}, broad: >={MIN_BROAD_CATS} cats @ >={BROAD_RATE_MIN}")

    print("\n[load] D_probe qid -> (prompt_type, category) map")
    qid_to_type, qid_to_cat = load_qid_metadata()
    print(f"  n_qids={len(qid_to_type)}")

    summaries: dict[str, dict] = {}
    hashes: dict[str, str] = {}
    for spec in MODELS:
        tag = spec["tag"]
        infile = PROFILE_DIR / spec["input"]
        print(f"\n[{tag}] reading {infile.name}")
        df = pd.read_parquet(infile)
        print(f"  n_features={len(df)}, n_layers={df['layer'].nunique()}")

        out_df = classify(df)
        out_df = add_contamination_flags(out_df, qid_to_type, qid_to_cat)
        out_path = PROFILE_DIR / f"phase2_candidates_{tag}.parquet"
        out_df.to_parquet(out_path, index=False)
        print(f"  saved: {out_path.name} ({len(out_df)} rows)")
        hashes[tag] = sha256_of(out_path)

        s = summarize(out_df, tag)
        summaries[tag] = s
        print(f"  type counts: {s['type_counts']}")
        print(f"  type pct:    {s['type_counts_pct']}")
        print(f"  category_selective by dominant cat: {s['category_selective_by_dominant_cat']}")
        cont = s["contamination"]["control_frac_in_dom_cat_category_selective_only"]
        print(f"  contamination (cs, in dom cat): median={cont['median']}, mean={cont['mean']:.4f}, "
              f">baseline={cont['n_above_baseline']}, >warn={cont['n_above_warn']}, >severe={cont['n_above_severe']}")

    summary_path = PROFILE_DIR / "phase2_candidate_summary.json"
    payload = {
        "frozen_at": now_iso(),
        "thresholds": {
            "n_rare_threshold": N_RARE_THRESHOLD,
            "top1_rate_min": TOP1_RATE_MIN,
            "selectivity_min": SELECTIVITY_MIN,
            "min_broad_cats": MIN_BROAD_CATS,
            "broad_rate_min": BROAD_RATE_MIN,
        },
        "models": summaries,
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    print(f"\n[summary] {summary_path.name}")

    # Update hash_log
    log = json.loads(HASH_LOG.read_text()) if HASH_LOG.exists() else {}
    for tag, sha in hashes.items():
        log[f"phase2_candidates_{tag}_hash"] = {
            "file": str((PROFILE_DIR / f"phase2_candidates_{tag}.parquet").relative_to(REPO_ROOT)),
            "sha256": sha,
            "n_features": summaries[tag]["n_total_features"],
            "type_counts": summaries[tag]["type_counts"],
            "thresholds": payload["thresholds"],
            "frozen_at": now_iso(),
        }
    log["phase2_candidate_summary_hash"] = {
        "file": str(summary_path.relative_to(REPO_ROOT)),
        "sha256": sha256_of(summary_path),
        "frozen_at": now_iso(),
    }
    HASH_LOG.write_text(json.dumps(log, indent=2, ensure_ascii=False, sort_keys=True))
    print(f"[done] hash_log updated")


if __name__ == "__main__":
    main()
