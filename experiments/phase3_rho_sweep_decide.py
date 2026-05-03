"""Phase 3 v001.1 — C1-C5 採否判定 (separated from data generation).

Pre-registered design: docs/preregistrations/phase3_v001_1_rho_sweep.md (v2)

Reads:
  data/profiling/v001/phase3_rho_sweep_validation.json
  data/profiling/v001/phase3_causal_light_pythia_160m_rho{rho_tag}.parquet × 3

Computes:
  C1 (transparency only) per-ρ Pearson(linear, real)
  C2 同一 pair の REAL TE 値: 3 ρ 全て同符号 pair 比率 ≥ 75%
  C3 REAL TE の 3 ρ pairwise Spearman 最小値 ≥ 0.80
  C4 per-ρ Pearson 安定性 max - min ≤ 0.10
  C5 少なくとも 1 ρ で Pearson ≥ 0.85

Decision matrix (pre-registered §3.5):
  C2-C5 all pass        → ALLOW Phase 4 entry; ρ=0.5 採用
  C5 only fail          → FREEZE; escalate to IG (v001.2)
  C4 fail               → FREEZE; ρ-dependent → IG or narrow ρ range
  C2 or C3 fail         → FREEZE; real effect unstable → review candidate rules

Writes:
  data/profiling/v001/phase3_rho_sweep_decision.json
  Updates fallback_reason column on 3 parquets
  configs/hash_log.json updated
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "profiling" / "v001"
VALIDATION_PATH = OUT_DIR / "phase3_rho_sweep_validation.json"
DECISION_PATH = OUT_DIR / "phase3_rho_sweep_decision.json"
HASH_LOG = REPO_ROOT / "configs" / "hash_log.json"

# Thresholds (pre-registered §3.3)
C2_SIGN_AGREEMENT_MIN = 0.75
C3_SPEARMAN_MIN = 0.80
C4_PEARSON_SPREAD_MAX = 0.10
C5_PEARSON_MIN = 0.85
REAL_FLOOR = 1e-3  # |real_d| < FLOOR pairs excluded from sign-agreement

# Pre-registered §3.4: ρ=0.5 採用
RHO_PHASE4_INPUT = 0.5


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rho_tag(rho: float) -> str:
    return f"rho{int(round(rho * 100)):03d}"


def out_parquet_for_rho(rho: float) -> Path:
    return OUT_DIR / f"phase3_causal_light_pythia_160m_{rho_tag(rho)}.parquet"


def compute_c_metrics(validation: dict) -> dict:
    rho_sweep = validation["rho_sweep"]
    rho_keys = [str(r) for r in rho_sweep]
    per_rho = validation["per_rho"]

    # C1 (transparency only)
    c1 = {r: per_rho[r]["pearson_r"] for r in rho_keys}

    # Build per-pair real TE matrix: shape (n_pairs, n_rho)
    real_arrays = [np.array(validation["real_te_per_pair_per_rho"][r], dtype=float) for r in rho_keys]
    n_pairs = min(len(a) for a in real_arrays) if real_arrays else 0
    real_matrix = np.stack([a[:n_pairs] for a in real_arrays], axis=1) if n_pairs else np.zeros((0, len(rho_keys)))

    # C2: sign agreement on pairs where ALL ρ have |real| > REAL_FLOOR
    if n_pairs:
        nontrivial_mask = (np.abs(real_matrix) > REAL_FLOOR).all(axis=1)
        n_nontrivial = int(nontrivial_mask.sum())
        if n_nontrivial > 0:
            signs = np.sign(real_matrix[nontrivial_mask])
            all_same = (signs == signs[:, [0]]).all(axis=1)
            c2_agreement = float(all_same.mean())
        else:
            c2_agreement = float("nan")
    else:
        n_nontrivial = 0
        c2_agreement = float("nan")

    # C3: pairwise Spearman of REAL TE arrays (use all pairs, not just nontrivial)
    pair_indices = []
    for i in range(len(rho_keys)):
        for j in range(i + 1, len(rho_keys)):
            pair_indices.append((i, j))
    pairwise_spearman: dict[str, float] = {}
    if n_pairs > 2:
        for i, j in pair_indices:
            rho_i = rho_keys[i]
            rho_j = rho_keys[j]
            corr = spearmanr(real_matrix[:, i], real_matrix[:, j])
            pairwise_spearman[f"{rho_i}_vs_{rho_j}"] = float(corr.statistic)
    c3_min = min(pairwise_spearman.values()) if pairwise_spearman else float("nan")

    # C4: Pearson spread
    pearson_values = [per_rho[r]["pearson_r"] for r in rho_keys
                      if not (per_rho[r]["pearson_r"] != per_rho[r]["pearson_r"])]  # filter NaN
    if pearson_values:
        c4_spread = float(max(pearson_values) - min(pearson_values))
    else:
        c4_spread = float("nan")

    # C5: max Pearson
    if pearson_values:
        c5_max = float(max(pearson_values))
    else:
        c5_max = float("nan")

    return {
        "C1_per_rho_pearson": c1,
        "C2_sign_agreement": {
            "value": c2_agreement,
            "n_nontrivial_pairs": n_nontrivial,
            "n_pairs_total": int(n_pairs),
            "real_floor": REAL_FLOOR,
            "threshold": C2_SIGN_AGREEMENT_MIN,
            "pass": bool(c2_agreement == c2_agreement and c2_agreement >= C2_SIGN_AGREEMENT_MIN),
        },
        "C3_pairwise_spearman": {
            "values": pairwise_spearman,
            "min": c3_min,
            "threshold": C3_SPEARMAN_MIN,
            "pass": bool(c3_min == c3_min and c3_min >= C3_SPEARMAN_MIN),
        },
        "C4_pearson_spread": {
            "value": c4_spread,
            "per_rho": c1,
            "threshold": C4_PEARSON_SPREAD_MAX,
            "pass": bool(c4_spread == c4_spread and c4_spread <= C4_PEARSON_SPREAD_MAX),
        },
        "C5_max_pearson": {
            "value": c5_max,
            "threshold": C5_PEARSON_MIN,
            "pass": bool(c5_max == c5_max and c5_max >= C5_PEARSON_MIN),
        },
    }


def decide(c_metrics: dict) -> dict:
    """Pre-registered §3.5 採否マトリクス。"""
    c2_pass = c_metrics["C2_sign_agreement"]["pass"]
    c3_pass = c_metrics["C3_pairwise_spearman"]["pass"]
    c4_pass = c_metrics["C4_pearson_spread"]["pass"]
    c5_pass = c_metrics["C5_max_pearson"]["pass"]

    if c2_pass and c3_pass and c4_pass and c5_pass:
        status = "ALLOW"
        fallback_reason = None
        next_action = f"Phase 4 入力に ρ={RHO_PHASE4_INPUT} の parquet を採用 (pre-registered)"
    elif (not c2_pass) or (not c3_pass):
        status = "FREEZE_C2_OR_C3_FAIL"
        fallback_reason = "real_unstable_review_candidates"
        next_action = "Phase 4 凍結。candidate 規則 (Phase 2) 見直しを v001.2 で検討"
    elif not c4_pass:
        status = "FREEZE_C4_FAIL"
        fallback_reason = "rho_dependent_escalate_IG"
        next_action = "Phase 4 凍結。線形近似精度が ρ 強依存 → IG 切替を v001.2 で検討"
    elif not c5_pass:
        status = "FREEZE_C5_ONLY_FAIL"
        fallback_reason = "linear_approx_inadequate_escalate_IG"
        next_action = "Phase 4 凍結。線形近似が本 setup で不適 → IG (Marks 2024 方式) 切替を v001.2 で実施"
    else:
        status = "UNKNOWN"
        fallback_reason = "unknown"
        next_action = "ロジック分岐の確認が必要"

    return {
        "status": status,
        "fallback_reason": fallback_reason,
        "next_action": next_action,
        "C_pass": {
            "C2": c2_pass, "C3": c3_pass, "C4": c4_pass, "C5": c5_pass,
        },
    }


def update_parquets_fallback(rho_sweep: list[float], fallback_reason: str | None) -> dict[float, str]:
    """Update fallback_reason column on each parquet. Returns sha256 per ρ."""
    sha256s: dict[float, str] = {}
    for rho in rho_sweep:
        out = out_parquet_for_rho(rho)
        df = pd.read_parquet(out)
        df["fallback_reason"] = fallback_reason
        df.to_parquet(out, index=False)
        sha256s[rho] = sha256_of(out)
    return sha256s


def update_hash_log(rho_sweep: list[float], decision: dict, sha256s: dict[float, str]) -> None:
    log = json.loads(HASH_LOG.read_text()) if HASH_LOG.exists() else {}
    for rho in rho_sweep:
        key = f"phase3_causal_light_pythia_{rho_tag(rho)}_hash"
        existing = log.get(key, {})
        existing["sha256"] = sha256s[rho]
        existing["fallback_reason"] = decision["fallback_reason"]
        existing["decision_status"] = decision["status"]
        existing["frozen_at"] = now_iso()
        log[key] = existing
    log["phase3_rho_sweep_decision_hash"] = {
        "file": str(DECISION_PATH.relative_to(REPO_ROOT)),
        "sha256": sha256_of(DECISION_PATH),
        "decision_status": decision["status"],
        "C_pass": decision["C_pass"],
        "frozen_at": now_iso(),
    }
    HASH_LOG.write_text(json.dumps(log, indent=2, ensure_ascii=False, sort_keys=True))


def main() -> None:
    print("--- Phase 3 v001.1 C1-C5 採否判定 ---")
    print(f"  pre-registration: docs/preregistrations/phase3_v001_1_rho_sweep.md")
    print(f"  validation: {VALIDATION_PATH.name}")

    val = json.loads(VALIDATION_PATH.read_text())
    rho_sweep = val["rho_sweep"]
    print(f"  rho_sweep: {rho_sweep}")
    print(f"  n_pairs (all ρ succeed): {val['n_validation_pairs_used_all_rho_succeed']}")

    c_metrics = compute_c_metrics(val)

    print()
    print("[C1 transparency only] per-ρ Pearson(linear, real):")
    for rho_str, p in c_metrics["C1_per_rho_pearson"].items():
        print(f"  ρ={rho_str}: {p:.4f}")

    print()
    print(f"[C2] same-sign rate (REAL TE, |real|>{REAL_FLOOR}, all 3 ρ same sign):")
    c2 = c_metrics["C2_sign_agreement"]
    print(f"  value={c2['value']:.4f}  threshold≥{c2['threshold']}  "
          f"n_nontrivial={c2['n_nontrivial_pairs']}/{c2['n_pairs_total']}  "
          f"pass={c2['pass']}")

    print()
    print(f"[C3] pairwise Spearman of REAL TE values:")
    c3 = c_metrics["C3_pairwise_spearman"]
    for k, v in c3["values"].items():
        print(f"  {k}: {v:.4f}")
    print(f"  min={c3['min']:.4f}  threshold≥{c3['threshold']}  pass={c3['pass']}")

    print()
    print(f"[C4] Pearson(linear, real) spread:")
    c4 = c_metrics["C4_pearson_spread"]
    print(f"  spread={c4['value']:.4f}  threshold≤{c4['threshold']}  pass={c4['pass']}")

    print()
    print(f"[C5] max Pearson(linear, real) across ρ:")
    c5 = c_metrics["C5_max_pearson"]
    print(f"  max={c5['value']:.4f}  threshold≥{c5['threshold']}  pass={c5['pass']}")

    decision = decide(c_metrics)
    print()
    print(f"[decision] status={decision['status']}")
    print(f"  fallback_reason: {decision['fallback_reason']}")
    print(f"  next_action: {decision['next_action']}")

    payload = {
        "pre_registration_path": "docs/preregistrations/phase3_v001_1_rho_sweep.md",
        "rho_sweep": rho_sweep,
        "rho_phase4_input_committed": RHO_PHASE4_INPUT,
        "n_validation_pairs_used": val["n_validation_pairs_used_all_rho_succeed"],
        "thresholds_pre_committed": {
            "C2_sign_agreement_min": C2_SIGN_AGREEMENT_MIN,
            "C3_spearman_min": C3_SPEARMAN_MIN,
            "C4_pearson_spread_max": C4_PEARSON_SPREAD_MAX,
            "C5_pearson_min": C5_PEARSON_MIN,
        },
        "metrics": c_metrics,
        "decision": decision,
        "frozen_at": now_iso(),
    }
    DECISION_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n[saved] {DECISION_PATH.name}")

    print(f"\n[updating parquets] fallback_reason -> {decision['fallback_reason']!r}")
    sha256s = update_parquets_fallback(rho_sweep, decision["fallback_reason"])
    for rho, sha in sha256s.items():
        print(f"  ρ={rho}: sha256={sha[:16]}...")

    update_hash_log(rho_sweep, decision, sha256s)
    print(f"\n[done] hash_log updated")


if __name__ == "__main__":
    main()
