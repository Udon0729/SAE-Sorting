"""Freeze popularity_bucket_definition for D_fact_v001 (PopQA portion).

Outputs:
  - data/raw/popqa/popqa_test.parquet           : frozen PopQA snapshot
  - configs/popularity_bucket_definition.yaml   : bucket boundaries + metadata
  - configs/hash_log.json                       : SHA256 of both artifacts

Bucket convention:
  low : log10(s_pop) <= q33
  mid : q33 < log10(s_pop) <= q67
  high: q67 < log10(s_pop)
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = REPO_ROOT / "data" / "raw" / "popqa"
CONFIGS = REPO_ROOT / "configs"
HF_CACHE = REPO_ROOT / "data" / "raw" / "popqa_cache"

POPQA_SOURCE = "akariasai/PopQA"
SPLIT = "test"
FREEZE_VERSION = "v001"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    CONFIGS.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(POPQA_SOURCE, cache_dir=str(HF_CACHE))[SPLIT]
    n = len(ds)
    df = ds.to_pandas()

    snapshot_path = DATA_RAW / f"popqa_{SPLIT}.parquet"
    df.to_parquet(snapshot_path, index=False)

    s_pop = df["s_pop"].to_numpy(dtype=np.float64)
    assert (s_pop > 0).all(), "s_pop must be strictly positive for log10"
    log_pop = np.log10(s_pop)

    q33 = float(np.quantile(log_pop, 1.0 / 3.0))
    q67 = float(np.quantile(log_pop, 2.0 / 3.0))
    assert q33 < q67, f"degenerate quantiles: q33={q33} q67={q67}"

    buckets = np.where(log_pop <= q33, "low", np.where(log_pop <= q67, "mid", "high"))
    n_low = int((buckets == "low").sum())
    n_mid = int((buckets == "mid").sum())
    n_high = int((buckets == "high").sum())
    assert n_low + n_mid + n_high == n

    frozen_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    bucket_def = {
        "name": "popularity_bucket_definition",
        "version": FREEZE_VERSION,
        "frozen_at": frozen_at,
        "score_formula": "log10(s_pop)",
        "source": {
            "dataset": POPQA_SOURCE,
            "split": SPLIT,
            "n_rows": n,
            "snapshot": str(snapshot_path.relative_to(REPO_ROOT)),
        },
        "boundaries": {
            "low_max": q33,
            "mid_max": q67,
        },
        "assignment_rule": (
            "low if log10(s_pop) <= low_max; "
            "mid if low_max < log10(s_pop) <= mid_max; "
            "high otherwise"
        ),
        "bucket_counts": {"low": n_low, "mid": n_mid, "high": n_high},
        "out_of_scope_value": "n/a",
        "out_of_scope_datasets": ["LAMA", "ParaRel", "CounterFact", "D_synth"],
    }

    bucket_yaml_path = CONFIGS / "popularity_bucket_definition.yaml"
    with bucket_yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(bucket_def, f, sort_keys=False, allow_unicode=True)

    snapshot_sha = sha256_of(snapshot_path)
    bucket_yaml_sha = sha256_of(bucket_yaml_path)

    hash_log_path = CONFIGS / "hash_log.json"
    if hash_log_path.exists():
        with hash_log_path.open("r", encoding="utf-8") as f:
            hash_log = json.load(f)
    else:
        hash_log = {}

    hash_log["popqa_test_snapshot_hash"] = {
        "file": str(snapshot_path.relative_to(REPO_ROOT)),
        "sha256": snapshot_sha,
        "n_rows": n,
        "source": POPQA_SOURCE,
        "frozen_at": frozen_at,
    }
    hash_log["popularity_bucket_definition_hash"] = {
        "file": str(bucket_yaml_path.relative_to(REPO_ROOT)),
        "sha256": bucket_yaml_sha,
        "frozen_at": frozen_at,
    }

    with hash_log_path.open("w", encoding="utf-8") as f:
        json.dump(hash_log, f, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"snapshot     : {snapshot_path}  sha256={snapshot_sha}")
    print(f"bucket yaml  : {bucket_yaml_path}  sha256={bucket_yaml_sha}")
    print(f"q33={q33:.10f} q67={q67:.10f}")
    print(f"counts: low={n_low} mid={n_mid} high={n_high} (n={n})")
    print(f"shares: low={n_low/n:.4f} mid={n_mid/n:.4f} high={n_high/n:.4f}")


if __name__ == "__main__":
    main()
