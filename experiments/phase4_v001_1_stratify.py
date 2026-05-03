"""Phase 4 v001.1 — stratified feature selection (8 strata × 30 = 240).

Pre-registered design: docs/preregistrations/phase4_v001_1_measurement.md §3.4 + §3.5

Inputs:
  data/profiling/v001/phase3_causal_light_pythia_160m_rho050_v3.parquet  # per-prompt logP
  data/profiling/v001/phase4_cluster_assignment.parquet                  # cluster_id (S5)
  data/profiling/v001/phase2_candidates_pythia_160m.parquet              # candidate_type (S6/S7)

Per-feature aggregates from v3 (probe rows, split_role == 'fit'):
  per_cat_mean_delta[c]  = mean over fit prompts of cat c of delta_logP_attribution
  damage_per_cat[c]      = -per_cat_mean_delta[c]            # §3.1 sign convention
  TE_max                 = max_c damage_per_cat[c]
  primary_cat            = argmax_c damage_per_cat[c]
  Specificity            = TE_max - max_{c'≠primary} damage_per_cat[c']
  UD                     = mean over utility prompts of (-delta_logP_attribution / n_target_tokens)

NaN handling (§3.4):
  pool = features where all 4 cats have non-NaN per_cat_mean_delta
  if |pool| < 500: fallback to any-cat-non-NaN pool + 4 binary missingness mask cols

Strata (priority order, dedup against higher strata):
  S1: top 30 by |TE_max|                                   (Phase 3 v3 fit)
  S2: top 30 by Specificity                                (Phase 3 v3 fit)
  S3: top 30 by smallest |UD|                              (Phase 3 v3 utility)
  S4: top 30 of Pareto frontier (TE_max desc, UD asc) by TE_max desc
  S5: random 30 from cluster_id == -1                      (Phase 4 v001)
  S6: random 30 from candidate_type == 'broad'             (Phase 2)
  S7: random 30 from candidate_type == 'category_selective' (Phase 2)
  S8: random 30 from pool (uniform)

random_state: np.random.default_rng(42)

Outputs:
  data/profiling/v001/phase4_v001_1_stratum_assignments.parquet  # 240 rows (or less)
  data/profiling/v001/phase4_v001_1_stratum_summary.json         # counts + ranges + fallback flag
  configs/hash_log.json updated (2 entries)
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "profiling" / "v001"
V3_PARQUET = DATA_DIR / "phase3_causal_light_pythia_160m_rho050_v3.parquet"
CLUSTER_PARQUET = DATA_DIR / "phase4_cluster_assignment.parquet"
CAND_PARQUET = DATA_DIR / "phase2_candidates_pythia_160m.parquet"
OUT_ASSIGN = DATA_DIR / "phase4_v001_1_stratum_assignments.parquet"
OUT_SUMMARY = DATA_DIR / "phase4_v001_1_stratum_summary.json"
HASH_LOG = REPO_ROOT / "configs" / "hash_log.json"

CATS = ["person_attribute", "geography", "organization", "occupation"]
N_PER_STRATUM = 30
NAN_FALLBACK_THRESHOLD = 500
RNG_SEED = 42


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_per_feature_aggregates(v3: pd.DataFrame) -> pd.DataFrame:
    """From v3 long format compute per-(layer, feature) aggregates used by S1-S4."""
    probe_fit = v3[(v3["source"] == "probe") & (v3["split_role"] == "fit")]
    util = v3[v3["source"] == "utility"].copy()

    # per-cat mean delta over fit-split probe rows
    per_cat_long = (
        probe_fit.groupby(["layer", "feature_idx", "category"])["delta_logP_attribution"]
        .mean()
        .reset_index()
    )
    per_cat = per_cat_long.pivot(
        index=["layer", "feature_idx"], columns="category", values="delta_logP_attribution"
    )
    # Ensure all 4 cat columns present
    for c in CATS:
        if c not in per_cat.columns:
            per_cat[c] = float("nan")
    per_cat = per_cat[CATS]
    per_cat.columns = [f"per_cat_mean_delta_{c}" for c in CATS]

    # damage = -delta (§3.1 sign convention)
    damage = -per_cat.copy()
    damage.columns = [f"damage_{c}" for c in CATS]

    # TE_max, primary_cat, Specificity (NaN-aware)
    damage_arr = damage.to_numpy()  # shape (N, 4)
    te_max = np.nanmax(damage_arr, axis=1)
    primary_idx = np.nanargmax(np.where(np.isnan(damage_arr), -np.inf, damage_arr), axis=1)
    primary_cat = np.array([CATS[i] for i in primary_idx])
    # 2nd-best: mask primary then nanmax
    masked = damage_arr.copy()
    rows = np.arange(masked.shape[0])
    masked[rows, primary_idx] = np.nan
    second_max = np.nanmax(masked, axis=1)  # NaN if only one non-NaN cat
    specificity = te_max - second_max

    agg = damage.reset_index()
    agg["te_max_fit"] = te_max
    agg["primary_cat_fit"] = primary_cat
    agg["specificity_fit"] = specificity
    agg["all_4_cats_present"] = damage.notna().all(axis=1).to_numpy()

    # UD = mean over utility prompts of (-delta / n_target_tokens)
    util["damage_per_token"] = -util["delta_logP_attribution"] / util["n_target_tokens"].clip(lower=1)
    ud = (
        util.groupby(["layer", "feature_idx"])["damage_per_token"]
        .mean()
        .rename("ud")
        .reset_index()
    )
    agg = agg.merge(ud, on=["layer", "feature_idx"], how="left")

    return agg


def pareto_frontier_indices(te: np.ndarray, ud: np.ndarray) -> np.ndarray:
    """Return indices (into input arrays) of features on the Pareto frontier
    where 'better' = higher TE AND lower UD. Sort by TE desc, scan with
    running min UD; a feature is on the frontier iff its UD < running_min."""
    order = np.argsort(-te)  # TE descending
    running_min = np.inf
    frontier_mask = np.zeros(len(te), dtype=bool)
    for idx in order:
        if ud[idx] < running_min:
            frontier_mask[idx] = True
            running_min = ud[idx]
    return order[frontier_mask[order]]  # indices in TE-desc order


def assign_strata(agg: pd.DataFrame, cluster_df: pd.DataFrame, cand_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply 8 strata × 30 with priority dedup. Returns (long-format assignments, summary)."""
    rng = np.random.default_rng(RNG_SEED)

    # Join cluster_id and candidate_type for downstream filters / metadata
    agg = agg.merge(cluster_df[["layer", "feature_idx", "cluster_id"]], on=["layer", "feature_idx"], how="left")
    agg = agg.merge(cand_df[["layer", "feature_idx", "candidate_type"]], on=["layer", "feature_idx"], how="left")

    # NaN filter §3.4
    n_total = len(agg)
    n_all_4 = int(agg["all_4_cats_present"].sum())
    fallback_used = n_all_4 < NAN_FALLBACK_THRESHOLD
    if fallback_used:
        agg["any_cat_present"] = agg[[f"damage_{c}" for c in CATS]].notna().any(axis=1).to_numpy()
        pool_mask = agg["any_cat_present"].to_numpy()
        for c in CATS:
            agg[f"missing_mask_{c}"] = agg[f"damage_{c}"].isna().to_numpy()
    else:
        pool_mask = agg["all_4_cats_present"].to_numpy()

    pool_idx_set = set(np.where(pool_mask)[0].tolist())
    print(f"  NaN filter: total={n_total}, all_4_present={n_all_4}, fallback={fallback_used}, pool={len(pool_idx_set)}")

    assigned: set[int] = set()
    rows: list[dict] = []
    stratum_stats: dict[str, dict] = {}

    def take_top_k(candidate_idxs: list[int], k: int, rank_metric: np.ndarray, sort_descending: bool) -> list[int]:
        """Pick top-k by rank_metric, descending or ascending. NaN excluded."""
        cand = [i for i in candidate_idxs if not np.isnan(rank_metric[i])]
        cand.sort(key=lambda i: (-rank_metric[i] if sort_descending else rank_metric[i]))
        return cand[:k]

    def take_random_k(candidate_idxs: list[int], k: int) -> list[int]:
        if len(candidate_idxs) <= k:
            return candidate_idxs
        chosen = rng.choice(len(candidate_idxs), size=k, replace=False)
        return [candidate_idxs[i] for i in chosen]

    def record(stratum: str, picks: list[int], rank_metric_name: str, rank_values: np.ndarray | None) -> None:
        for r, idx in enumerate(picks, start=1):
            row = agg.iloc[idx]
            rows.append({
                "layer": int(row["layer"]),
                "feature_idx": int(row["feature_idx"]),
                "stratum": stratum,
                "rank_in_stratum": int(r),
                "stratum_metric_name": rank_metric_name,
                "stratum_metric_value": float(rank_values[idx]) if rank_values is not None else float("nan"),
                "te_max_fit": float(row["te_max_fit"]) if pd.notna(row["te_max_fit"]) else float("nan"),
                "specificity_fit": float(row["specificity_fit"]) if pd.notna(row["specificity_fit"]) else float("nan"),
                "ud": float(row["ud"]) if pd.notna(row["ud"]) else float("nan"),
                "primary_cat_fit": str(row["primary_cat_fit"]) if pd.notna(row["primary_cat_fit"]) else "",
                "damage_person_attribute": float(row["damage_person_attribute"]) if pd.notna(row["damage_person_attribute"]) else float("nan"),
                "damage_geography": float(row["damage_geography"]) if pd.notna(row["damage_geography"]) else float("nan"),
                "damage_organization": float(row["damage_organization"]) if pd.notna(row["damage_organization"]) else float("nan"),
                "damage_occupation": float(row["damage_occupation"]) if pd.notna(row["damage_occupation"]) else float("nan"),
                "candidate_type": str(row["candidate_type"]) if pd.notna(row["candidate_type"]) else "",
                "cluster_id_v001": int(row["cluster_id"]) if pd.notna(row["cluster_id"]) else -999,
                "all_4_cats_present": bool(row["all_4_cats_present"]),
            })
            assigned.add(idx)
        metric_arr = np.array([rank_values[i] for i in picks]) if rank_values is not None else np.array([])
        stratum_stats[stratum] = {
            "n_picked": len(picks),
            "n_target": N_PER_STRATUM,
            "rank_metric": rank_metric_name,
            "metric_min": float(np.min(metric_arr)) if metric_arr.size else None,
            "metric_max": float(np.max(metric_arr)) if metric_arr.size else None,
        }
        print(f"  {stratum}: {len(picks)}/{N_PER_STRATUM}  metric={rank_metric_name}"
              + (f"  range=[{np.min(metric_arr):.4g}, {np.max(metric_arr):.4g}]" if metric_arr.size else ""))

    # --- S1: top 30 by |TE_max| (fit) ---
    abs_te = np.abs(agg["te_max_fit"].to_numpy())
    s1_pool = sorted(pool_idx_set - assigned)
    s1 = take_top_k(s1_pool, N_PER_STRATUM, abs_te, sort_descending=True)
    record("S1", s1, "abs_te_max_fit", abs_te)

    # --- S2: top 30 by Specificity (fit) ---
    spec = agg["specificity_fit"].to_numpy()
    s2_pool = sorted(pool_idx_set - assigned)
    s2 = take_top_k(s2_pool, N_PER_STRATUM, spec, sort_descending=True)
    record("S2", s2, "specificity_fit", spec)

    # --- S3: bottom 30 by |UD| ---
    abs_ud = np.abs(agg["ud"].to_numpy())
    s3_pool = sorted(pool_idx_set - assigned)
    s3 = take_top_k(s3_pool, N_PER_STRATUM, abs_ud, sort_descending=False)
    record("S3", s3, "abs_ud_ascending", abs_ud)

    # --- S4: Pareto frontier (TE high, UD low), top 30 by TE desc ---
    te_arr = agg["te_max_fit"].to_numpy()
    ud_arr = agg["ud"].to_numpy()
    valid = (~np.isnan(te_arr)) & (~np.isnan(ud_arr)) & pool_mask
    valid_idxs = np.where(valid)[0]
    frontier_local = pareto_frontier_indices(te_arr[valid_idxs], ud_arr[valid_idxs])
    frontier_global = valid_idxs[frontier_local]  # already in TE-desc order
    s4 = [int(i) for i in frontier_global if int(i) not in assigned][:N_PER_STRATUM]
    record("S4", s4, "te_max_fit_pareto", te_arr)

    # --- S5: random 30 from cluster_id == -1 ---
    is_noise = agg["cluster_id"].to_numpy() == -1
    s5_pool = [i for i in np.where(is_noise & pool_mask)[0] if int(i) not in assigned]
    s5 = sorted(take_random_k(s5_pool, N_PER_STRATUM))
    record("S5", s5, "random_cluster_noise", None)

    # --- S6: random 30 from candidate_type == 'broad' ---
    cand_type_arr = agg["candidate_type"].to_numpy()
    is_broad = cand_type_arr == "broad"
    s6_pool = [i for i in np.where(is_broad & pool_mask)[0] if int(i) not in assigned]
    s6 = sorted(take_random_k(s6_pool, N_PER_STRATUM))
    record("S6", s6, "random_broad", None)

    # --- S7: random 30 from candidate_type == 'category_selective' ---
    is_cs = cand_type_arr == "category_selective"
    s7_pool = [i for i in np.where(is_cs & pool_mask)[0] if int(i) not in assigned]
    s7 = sorted(take_random_k(s7_pool, N_PER_STRATUM))
    record("S7", s7, "random_category_selective", None)

    # --- S8: random 30 from full pool (uniform control) ---
    s8_pool = [i for i in np.where(pool_mask)[0] if int(i) not in assigned]
    s8 = sorted(take_random_k(s8_pool, N_PER_STRATUM))
    record("S8", s8, "random_uniform", None)

    df_out = pd.DataFrame(rows)
    df_out["layer"] = df_out["layer"].astype("int32")
    df_out["feature_idx"] = df_out["feature_idx"].astype("int32")
    df_out["rank_in_stratum"] = df_out["rank_in_stratum"].astype("int32")
    df_out["cluster_id_v001"] = df_out["cluster_id_v001"].astype("int32")

    summary = {
        "n_total_features_in_pool_v3": n_total,
        "n_features_all_4_cats_present": n_all_4,
        "nan_fallback_threshold": NAN_FALLBACK_THRESHOLD,
        "nan_fallback_used": bool(fallback_used),
        "n_features_in_analysis_pool": int(pool_mask.sum()),
        "rng_seed": RNG_SEED,
        "n_per_stratum_target": N_PER_STRATUM,
        "n_target_total": 8 * N_PER_STRATUM,
        "n_assigned_total": len(rows),
        "all_strata_full": all(s["n_picked"] == N_PER_STRATUM for s in stratum_stats.values()),
        "stratum_stats": stratum_stats,
        "frozen_at": now_iso(),
    }

    return df_out, summary


def update_hash_log() -> None:
    log = json.loads(HASH_LOG.read_text()) if HASH_LOG.exists() else {}
    log["phase4_v001_1_stratum_assignments_hash"] = {
        "file": str(OUT_ASSIGN.relative_to(REPO_ROOT)),
        "sha256": sha256_of(OUT_ASSIGN),
        "depends_on": [
            "phase3_causal_light_pythia_rho050_v3_hash",
            "phase4_cluster_assignment_hash",
            "phase2_candidates_pythia_160m_hash",
        ],
        "rng_seed": RNG_SEED,
        "n_per_stratum_target": N_PER_STRATUM,
        "frozen_at": now_iso(),
    }
    log["phase4_v001_1_stratum_summary_hash"] = {
        "file": str(OUT_SUMMARY.relative_to(REPO_ROOT)),
        "sha256": sha256_of(OUT_SUMMARY),
        "frozen_at": now_iso(),
    }
    HASH_LOG.write_text(json.dumps(log, indent=2, ensure_ascii=False, sort_keys=True))


def main() -> None:
    print("--- Phase 4 v001.1 stratify (8 × 30 = 240) ---")

    print(f"[load] {V3_PARQUET.name}")
    v3 = pd.read_parquet(V3_PARQUET)
    print(f"  rows={len(v3)}, unique (layer, feature_idx)={v3.groupby(['layer','feature_idx']).ngroups}")

    print(f"[load] {CLUSTER_PARQUET.name}")
    cluster_df = pd.read_parquet(CLUSTER_PARQUET)

    print(f"[load] {CAND_PARQUET.name}")
    cand_df = pd.read_parquet(CAND_PARQUET)
    cand_df = cand_df[cand_df["candidate_type"].isin(["category_selective", "broad"])][
        ["layer", "feature_idx", "candidate_type"]
    ].reset_index(drop=True)

    print("\n[stage 1] per-feature aggregates from v3 (probe fit + utility)")
    agg = build_per_feature_aggregates(v3)
    print(f"  aggregated: {len(agg)} (layer, feature_idx) rows")

    print("\n[stage 2] strata assignment (priority dedup)")
    df_out, summary = assign_strata(agg, cluster_df, cand_df)
    print(f"\n  total assigned: {len(df_out)}/{8 * N_PER_STRATUM}")
    print(f"  all_strata_full: {summary['all_strata_full']}")

    OUT_ASSIGN.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(OUT_ASSIGN, index=False)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n  saved: {OUT_ASSIGN.name} ({OUT_ASSIGN.stat().st_size / 1024:.1f} KiB)")
    print(f"  saved: {OUT_SUMMARY.name}")

    update_hash_log()
    print(f"  hash_log updated: phase4_v001_1_stratum_assignments_hash + summary_hash")
    print("\n[done] Phase 4 v001.1 heavy script の入力として使える")


if __name__ == "__main__":
    main()
