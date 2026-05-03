"""Phase 4 v001 — Acceptance gate decision (G1-G5).

Pre-registered design: docs/preregistrations/phase4_v001.md §3.7

Inputs:
  data/profiling/v001/phase4_cluster_assignment.parquet
  data/profiling/v001/phase4_cluster_heavy_profile.parquet
  data/profiling/v001/phase4_cluster_labels.parquet
  data/profiling/v001/phase4_feature_vectors.parquet
  data/profiling/v001/phase3_causal_light_pythia_160m_rho050_v2.parquet (for G5 light reference)

Output:
  data/profiling/v001/phase4_decision.json
  hash_log entries updated with decision_status
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "profiling" / "v001"
ASSIGN = DATA_DIR / "phase4_cluster_assignment.parquet"
HEAVY = DATA_DIR / "phase4_cluster_heavy_profile.parquet"
LABELS = DATA_DIR / "phase4_cluster_labels.parquet"
FV = DATA_DIR / "phase4_feature_vectors.parquet"
PHASE3_V2 = DATA_DIR / "phase3_causal_light_pythia_160m_rho050_v2.parquet"
OUT = DATA_DIR / "phase4_decision.json"
HASH_LOG = REPO_ROOT / "configs" / "hash_log.json"

CATS = ["person_attribute", "geography", "organization", "occupation"]

G1_MIN_CLUSTERS = 5
G2_MIN_COVERAGE = 0.60
G3_MIN_KNOWLEDGE = 1
G5_MIN_PEARSON = 0.7


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    print("--- Phase 4 v001 — G1-G5 decision ---")
    assn = pd.read_parquet(ASSIGN)
    heavy = pd.read_parquet(HEAVY)
    labels = pd.read_parquet(LABELS)
    fv = pd.read_parquet(FV)
    phase3 = pd.read_parquet(PHASE3_V2)

    n_total = int(len(assn))
    valid = assn[assn["cluster_id"] >= 0]
    n_clusters = int(valid["cluster_id"].nunique()) if len(valid) else 0

    # G1: cluster count
    g1_pass = bool(n_clusters >= G1_MIN_CLUSTERS)
    print(f"  G1: clusters={n_clusters}  threshold={G1_MIN_CLUSTERS}  pass={g1_pass}")

    # G2: labeled coverage (UNKNOWN 以外 cluster の member 数 / 全 features)
    if len(labels):
        non_unknown_ids = labels[labels["label"] != "UNKNOWN"]["cluster_id"].astype(int).tolist()
        labeled_members = int(assn[assn["cluster_id"].isin(non_unknown_ids)].shape[0])
    else:
        labeled_members = 0
    g2_coverage = labeled_members / n_total if n_total else 0.0
    g2_pass = bool(g2_coverage >= G2_MIN_COVERAGE)
    print(f"  G2: labeled_coverage={g2_coverage:.4f}  threshold={G2_MIN_COVERAGE}  pass={g2_pass}")

    # G3: KNOWLEDGE-ASSOCIATED cluster count
    n_knowledge = int((labels["label"] == "KNOWLEDGE-ASSOCIATED").sum()) if len(labels) else 0
    g3_pass = bool(n_knowledge >= G3_MIN_KNOWLEDGE)
    print(f"  G3: KNOWLEDGE-ASSOCIATED clusters={n_knowledge}  threshold={G3_MIN_KNOWLEDGE}  pass={g3_pass}")

    # G4: median(te_max) of KNOWLEDGE > median(te_max) of UNKNOWN
    if len(labels):
        ka_te = labels[labels["label"] == "KNOWLEDGE-ASSOCIATED"]["te_max"]
        un_te = labels[labels["label"] == "UNKNOWN"]["te_max"]
        if len(ka_te) and len(un_te):
            g4_pass = bool(float(ka_te.median()) > float(un_te.median()))
            g4_detail = {"knowledge_te_median": float(ka_te.median()), "unknown_te_median": float(un_te.median())}
        else:
            g4_pass = False
            g4_detail = {"knowledge_te_median": None if not len(ka_te) else float(ka_te.median()),
                         "unknown_te_median": None if not len(un_te) else float(un_te.median())}
    else:
        g4_pass = False
        g4_detail = {}
    print(f"  G4: KNOWLEDGE_te_median > UNKNOWN_te_median  pass={g4_pass}  detail={g4_detail}")

    # G5: cluster-mean te_max (heavy H4 ρ=0.5) vs cluster-mean c_f[primary] (light)
    if len(heavy) and len(labels):
        h4 = heavy[heavy["intervention_id"] == "H4"]
        h4_idx = h4.set_index("cluster_id")[["te_dprobe", "primary_category"]]
        # light c_f[primary] per feature → mean per cluster
        merged = assn.merge(phase3, on=["layer", "feature_idx"], how="inner")
        cf_cols = {c: f"delta_m_target_per_cat_{c}" for c in CATS}
        light_means = []
        heavy_vals = []
        for cid, sub in merged.groupby("cluster_id"):
            if cid < 0 or cid not in h4_idx.index:
                continue
            primary = h4_idx.loc[cid, "primary_category"]
            cf_col = cf_cols[primary]
            light_means.append(float(sub[cf_col].mean()))
            heavy_vals.append(float(h4_idx.loc[cid, "te_dprobe"]))
        if len(light_means) >= 3:
            r = float(pearsonr(light_means, heavy_vals).statistic)
        else:
            r = float("nan")
        g5_pass = bool(np.isfinite(r) and r >= G5_MIN_PEARSON)
        g5_detail = {"pearson_r": r, "n_clusters_compared": len(light_means)}
    else:
        g5_pass = False
        g5_detail = {"pearson_r": None, "n_clusters_compared": 0}
    print(f"  G5: heavy-vs-light Pearson  pass={g5_pass}  detail={g5_detail}")

    gates = {
        "G1": {"pass": g1_pass, "value": n_clusters, "threshold": G1_MIN_CLUSTERS},
        "G2": {"pass": g2_pass, "value": g2_coverage, "threshold": G2_MIN_COVERAGE},
        "G3": {"pass": g3_pass, "value": n_knowledge, "threshold": G3_MIN_KNOWLEDGE},
        "G4": {"pass": g4_pass, **g4_detail},
        "G5": {"pass": g5_pass, **g5_detail, "threshold": G5_MIN_PEARSON},
    }
    failed = [k for k, v in gates.items() if not v["pass"]]
    if not failed:
        status = "ALLOW"
        next_action = "Phase 4 v001 完了。Phase 5 (cross-model 比較等) 検討可。"
    else:
        status = f"FREEZE_{'_'.join(failed)}_FAIL"
        next_action = "v001.1 として別 pre-reg を切り、対応する fallback (UMAP / threshold 緩和 / Phase 2 再評価) を検討。"

    decision = {
        "status": status,
        "gates": gates,
        "n_total_features": n_total,
        "n_clusters": n_clusters,
        "n_noise": int((assn["cluster_id"] == -1).sum()),
        "label_counts": labels["label"].value_counts().to_dict() if len(labels) else {},
        "next_action": next_action,
        "frozen_at": now_iso(),
    }
    OUT.write_text(json.dumps(decision, indent=2, ensure_ascii=False))
    print(f"\n  status: {status}")
    print(f"  next_action: {next_action}")
    print(f"  saved: {OUT.name}")

    # update hash_log: mark phase4 entries with decision_status
    log = json.loads(HASH_LOG.read_text()) if HASH_LOG.exists() else {}
    log["phase4_decision_hash"] = {
        "file": str(OUT.relative_to(REPO_ROOT)),
        "sha256": sha256_of(OUT),
        "status": status,
        "gates_pass": {k: v["pass"] for k, v in gates.items()},
        "frozen_at": now_iso(),
    }
    for tag in ["phase4_feature_vectors_hash", "phase4_cluster_assignment_hash",
                "phase4_top_activating_examples_hash", "phase4_cluster_heavy_profile_hash",
                "phase4_cluster_labels_hash"]:
        if tag in log:
            log[tag]["decision_status"] = status
            if failed:
                log[tag]["fallback_reason"] = f"phase4_v001_{status}"
            else:
                log[tag]["fallback_reason"] = None
    HASH_LOG.write_text(json.dumps(log, indent=2, ensure_ascii=False, sort_keys=True))
    print("  hash_log updated with decision_status")


if __name__ == "__main__":
    main()
