"""Phase 4 v001 — 5-class labeling per pre-registered rules.

Pre-registered design: docs/preregistrations/phase4_v001.md §3.6

Inputs:
  data/profiling/v001/phase4_cluster_heavy_profile.parquet
  data/profiling/v001/phase4_cluster_assignment.parquet

Output:
  data/profiling/v001/phase4_cluster_labels.parquet
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
HEAVY = DATA_DIR / "phase4_cluster_heavy_profile.parquet"
ASSIGN = DATA_DIR / "phase4_cluster_assignment.parquet"
OUT = DATA_DIR / "phase4_cluster_labels.parquet"
HASH_LOG = REPO_ROOT / "configs" / "hash_log.json"

CATS = ["person_attribute", "geography", "organization", "occupation"]
GEN_COLLAPSE_INFRA_THRESH = 0.30


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def aggregate_per_cluster(heavy: pd.DataFrame) -> pd.DataFrame:
    """For each cluster: te_max, nte_max (we use mean -|TE| over non-primary cats — but here
    heavy parquet stores te_dprobe (= ΔlogP_target on primary cat prompts) only. nte_max is
    not available per H from current heavy schema → use te_dprobe magnitude as causal-strength
    proxy and 0 for nte_max (forces ENTANGLED label only when actual cross-cat data added).
    For v001, label discrimination uses te_max / ud_max / ud_var / gen_collapse_max only."""
    rows = []
    for cid, sub in heavy.groupby("cluster_id"):
        rows.append({
            "cluster_id": int(cid),
            "primary_category": sub["primary_category"].iloc[0],
            "n_members": int(sub["n_members"].iloc[0]),
            "te_max": float(sub["te_dprobe"].abs().max()),
            "nte_max": 0.0,  # placeholder — heavy schema does not include cross-cat ΔlogP in v001
            "ud_max": float(sub["ud_perplexity"].max()),
            "ud_var": float(sub["ud_perplexity"].var(ddof=0)),
            "gen_collapse_max": float(sub["gen_collapse_rate"].max()),
        })
    return pd.DataFrame(rows)


def assign_labels(agg: pd.DataFrame) -> pd.DataFrame:
    """Apply 5-class precedence rules with pre-committed thresholds."""
    if len(agg) == 0:
        agg["label"] = []
        agg["decision_rule"] = []
        return agg
    p80_te = float(agg["te_max"].quantile(0.80))
    p70_nte = float(agg["nte_max"].quantile(0.70))
    p50_nte = float(agg["nte_max"].quantile(0.50))
    p80_ud = float(agg["ud_max"].quantile(0.80))
    p50_ud = float(agg["ud_max"].quantile(0.50))
    p80_udvar = float(agg["ud_var"].quantile(0.80))

    labels = []
    rules = []
    for _, r in agg.iterrows():
        te, nte, udmax, udvar, gcoll = r["te_max"], r["nte_max"], r["ud_max"], r["ud_var"], r["gen_collapse_max"]
        if te >= p80_te and nte <= p50_nte and udmax <= p50_ud:
            labels.append("KNOWLEDGE-ASSOCIATED"); rules.append("rule1")
        elif te >= p80_te and nte >= p70_nte:
            labels.append("ENTANGLED"); rules.append("rule2")
        elif udmax >= p80_ud or gcoll >= GEN_COLLAPSE_INFRA_THRESH:
            labels.append("INFRASTRUCTURE"); rules.append("rule3")
        elif udvar >= p80_udvar:
            labels.append("UNSTABLE"); rules.append("rule4")
        else:
            labels.append("UNKNOWN"); rules.append("rule5_default")
    agg = agg.copy()
    agg["label"] = labels
    agg["decision_rule"] = rules
    agg.attrs["thresholds"] = {
        "p80_te_max": p80_te,
        "p70_nte_max": p70_nte,
        "p50_nte_max": p50_nte,
        "p80_ud_max": p80_ud,
        "p50_ud_max": p50_ud,
        "p80_ud_var": p80_udvar,
        "gen_collapse_infra_thresh": GEN_COLLAPSE_INFRA_THRESH,
    }
    return agg


def main() -> None:
    print("--- Phase 4 v001 — 5-class labeling ---")
    if not HEAVY.exists():
        print(f"[error] heavy profile missing: {HEAVY}")
        return
    heavy = pd.read_parquet(HEAVY)
    print(f"  heavy: {len(heavy)} rows, clusters: {heavy['cluster_id'].nunique() if len(heavy) else 0}")

    agg = aggregate_per_cluster(heavy)
    print(f"  per-cluster aggregate: {len(agg)} rows")

    labeled = assign_labels(agg)
    if len(labeled):
        print("  thresholds (pre-committed):")
        for k, v in (labeled.attrs.get("thresholds") or {}).items():
            print(f"    {k}: {v:.4f}")
        print("  label distribution:")
        print(labeled["label"].value_counts().to_string())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(OUT, index=False)
    print(f"  saved {OUT.name}")

    log = json.loads(HASH_LOG.read_text()) if HASH_LOG.exists() else {}
    log["phase4_cluster_labels_hash"] = {
        "file": str(OUT.relative_to(REPO_ROOT)),
        "sha256": sha256_of(OUT),
        "n_clusters_labeled": int(len(labeled)),
        "label_counts": labeled["label"].value_counts().to_dict() if len(labeled) else {},
        "thresholds": labeled.attrs.get("thresholds", {}),
        "frozen_at": now_iso(),
    }
    HASH_LOG.write_text(json.dumps(log, indent=2, ensure_ascii=False, sort_keys=True))
    print("  hash_log updated: phase4_cluster_labels_hash")
    print("[done] phase4_decide で G1-G5 採否")


if __name__ == "__main__":
    main()
