"""Post-process Phase 3 v001 outputs to add rank-based acceptance interpretation.

Rationale: implementation.md fallback rule states that Pearson r < 0.85 with
Spearman ρ ≥ 0.80 is acceptable as rank-based result (downstream Phase 4
clustering uses rank/distance, not absolute calibration). The strict gate
(Pearson + Spearman + mean rel_err) was conservative; this script adds:

  validation JSON:
    accept_strict       : original gate (unchanged)
    accept_rank_based   : Spearman ρ ≥ 0.80  (rank-preserving)
    rel_err_median_filtered : median rel_err on |real| > floor pairs (robust to outliers)

  parquet:
    fallback_reason updated:
      None                                       if strict accept
      "rank_based_only_pearson_below_threshold"  if rank-based accept only
      "attribution_low_correlation"              if both fail

  hash_log:
    new flags surfaced; sha256 recomputed.

Idempotent: safe to re-run.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PARQUET = REPO_ROOT / "data/profiling/v001/phase3_causal_light_pythia_160m.parquet"
VALIDATION = REPO_ROOT / "data/profiling/v001/phase3_validation_pythia_160m.json"
HASH_LOG = REPO_ROOT / "configs/hash_log.json"

ACCEPT_PEARSON = 0.85
ACCEPT_SPEARMAN = 0.80
ACCEPT_REL_ERR_MEAN = 0.20
ACCEPT_REL_ERR_MEDIAN = 0.25  # slightly looser since median is robust


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    print("--- Phase 3 post-process (rank-based interpretation) ---")
    val = json.loads(VALIDATION.read_text())
    pearson = float(val["pearson_r"])
    spearman = float(val["spearman_rho"])
    rel_err_mean = float(val["rel_err_mean_filtered"])
    rel_err_median = float(val["rel_err_p50_filtered"])

    accept_strict = bool(
        pearson >= ACCEPT_PEARSON
        and spearman >= ACCEPT_SPEARMAN
        and rel_err_mean <= ACCEPT_REL_ERR_MEAN
    )
    accept_rank_based = bool(
        spearman >= ACCEPT_SPEARMAN
        and rel_err_median <= ACCEPT_REL_ERR_MEDIAN
    )

    if accept_strict:
        fallback = None
    elif accept_rank_based:
        fallback = "rank_based_only_pearson_below_threshold"
    else:
        fallback = "attribution_low_correlation"

    val["accept_strict"] = accept_strict
    val["accept_rank_based"] = accept_rank_based
    val["accept"] = accept_strict  # keep original semantics
    val["fallback_reason"] = fallback
    val["accept_rank_based_thresholds"] = {
        "spearman_rho_min": ACCEPT_SPEARMAN,
        "rel_err_median_filtered_max": ACCEPT_REL_ERR_MEDIAN,
    }
    val["postprocessed_at"] = now_iso()
    VALIDATION.write_text(json.dumps(val, indent=2))
    print(f"  validation: pearson={pearson:.4f} spearman={spearman:.4f} "
          f"rel_err_mean={rel_err_mean:.4f} rel_err_median={rel_err_median:.4f}")
    print(f"  accept_strict={accept_strict}  accept_rank_based={accept_rank_based}  fallback={fallback}")

    df = pd.read_parquet(PARQUET)
    df["fallback_reason"] = fallback
    df.to_parquet(PARQUET, index=False)
    print(f"  parquet: {len(df)} rows; fallback_reason -> {fallback!r}")

    log = json.loads(HASH_LOG.read_text()) if HASH_LOG.exists() else {}
    log["phase3_causal_light_pythia_hash"] = {
        **log.get("phase3_causal_light_pythia_hash", {}),
        "file": str(PARQUET.relative_to(REPO_ROOT)),
        "sha256": sha256_of(PARQUET),
        "n_candidates": int(len(df)),
        "fallback_reason": fallback,
        "frozen_at": now_iso(),
    }
    log["phase3_validation_pythia_hash"] = {
        **log.get("phase3_validation_pythia_hash", {}),
        "file": str(VALIDATION.relative_to(REPO_ROOT)),
        "sha256": sha256_of(VALIDATION),
        "accept_strict": accept_strict,
        "accept_rank_based": accept_rank_based,
        "fallback_reason": fallback,
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "rel_err_mean_filtered": rel_err_mean,
        "rel_err_median_filtered": rel_err_median,
        "frozen_at": now_iso(),
    }
    HASH_LOG.write_text(json.dumps(log, indent=2, ensure_ascii=False, sort_keys=True))
    print(f"  hash_log updated")
    print("[done]")


if __name__ == "__main__":
    main()
