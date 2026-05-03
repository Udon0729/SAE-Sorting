"""Freeze D_fact_v001: snapshot LAMA T-REx + CounterFact + ParaRel + PopQA.

PopQA portion already frozen by scripts/freeze_popularity_bucket_definition.py;
this script adds the remaining three sources and writes a unified manifest.

Sources (URLs and commit pinned for reproducibility):
  - LAMA T-REx (Petroni et al. 2019)
      url:  https://dl.fbaipublicfiles.com/LAMA/data.zip  (TREx subdir)
      raw:  data/raw/lama/data.zip
      out:  data/raw/lama/lama_trex.parquet
  - CounterFact (Meng et al. 2022)
      url:  https://rome.baulab.info/data/dsets/counterfact.json
      raw:  data/raw/counterfact/counterfact.json
      out:  data/raw/counterfact/counterfact.parquet
  - ParaRel (Elazar et al. 2021)
      url:  github yanaiela/pararel @ commit cb55546784...
      raw:  data/raw/pararel/pararel_<sha8>.tar.gz
      out:  data/raw/pararel/pararel_patterns.parquet

Outputs:
  - data/frozen/d_fact_v001/manifest.yaml
  - configs/hash_log.json (adds 4 hashes:
      lama_trex_snapshot_hash, counterfact_snapshot_hash,
      pararel_snapshot_hash, d_fact_v001_manifest_hash)
"""

from __future__ import annotations

import hashlib
import json
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = REPO_ROOT / "data" / "raw"
FROZEN = REPO_ROOT / "data" / "frozen" / "d_fact_v001"
CONFIGS = REPO_ROOT / "configs"

PARAREL_COMMIT = "cb5554678457beb5ac163d888f1ce8cf174b3f0b"
SOURCES = {
    "LAMA": {
        "url": "https://dl.fbaipublicfiles.com/LAMA/data.zip",
        "raw_subdir": "lama",
        "raw_file": "data.zip",
    },
    "CounterFact": {
        "url": "https://rome.baulab.info/data/dsets/counterfact.json",
        "raw_subdir": "counterfact",
        "raw_file": "counterfact.json",
    },
    "ParaRel": {
        "commit": PARAREL_COMMIT,
        "url": f"https://github.com/yanaiela/pararel/archive/{PARAREL_COMMIT}.tar.gz",
        "raw_subdir": "pararel",
        "raw_file": f"pararel_{PARAREL_COMMIT[:8]}.tar.gz",
    },
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  [skip download] {dest.name} present ({dest.stat().st_size} bytes)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [download] {url} -> {dest}")
    with urlopen(url) as r:
        dest.write_bytes(r.read())
    print(f"  [downloaded] {dest.stat().st_size} bytes")


def freeze_lama() -> dict:
    spec = SOURCES["LAMA"]
    raw = DATA_RAW / spec["raw_subdir"] / spec["raw_file"]
    download(spec["url"], raw)

    rows: list[dict] = []
    with zipfile.ZipFile(raw) as z:
        trex_names = [n for n in z.namelist() if n.startswith("data/TREx/") and n.endswith(".jsonl")]
        if not trex_names:
            raise RuntimeError("No data/TREx/*.jsonl files found in LAMA data.zip")
        for name in sorted(trex_names):
            relation_id = Path(name).stem
            with z.open(name) as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    row["relation_id_from_filename"] = relation_id
                    rows.append(row)

    df = pd.DataFrame(rows)
    out = DATA_RAW / spec["raw_subdir"] / "lama_trex.parquet"
    df.to_parquet(out, index=False)
    return {
        "name": "LAMA",
        "subset": "TREx",
        "source_url": spec["url"],
        "raw_archive": str(raw.relative_to(REPO_ROOT)),
        "raw_archive_sha256": sha256_of(raw),
        "snapshot": str(out.relative_to(REPO_ROOT)),
        "snapshot_sha256": sha256_of(out),
        "n_rows": len(df),
        "n_relations": int(df["relation_id_from_filename"].nunique()),
    }


def freeze_counterfact() -> dict:
    spec = SOURCES["CounterFact"]
    raw = DATA_RAW / spec["raw_subdir"] / spec["raw_file"]
    download(spec["url"], raw)

    with raw.open("r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.json_normalize(data)
    out = DATA_RAW / spec["raw_subdir"] / "counterfact.parquet"
    df.to_parquet(out, index=False)
    return {
        "name": "CounterFact",
        "source_url": spec["url"],
        "raw_archive": str(raw.relative_to(REPO_ROOT)),
        "raw_archive_sha256": sha256_of(raw),
        "snapshot": str(out.relative_to(REPO_ROOT)),
        "snapshot_sha256": sha256_of(out),
        "n_rows": len(df),
    }


def freeze_pararel() -> dict:
    spec = SOURCES["ParaRel"]
    raw = DATA_RAW / spec["raw_subdir"] / spec["raw_file"]
    download(spec["url"], raw)

    rows: list[dict] = []
    with tarfile.open(raw) as t:
        members = [
            m for m in t.getmembers()
            if "data/pattern_data/graphs_json/" in m.name and m.name.endswith(".jsonl")
        ]
        if not members:
            raise RuntimeError("No pararel pattern JSONL files found in tarball")
        for m in sorted(members, key=lambda x: x.name):
            relation_id = Path(m.name).stem
            f = t.extractfile(m)
            if f is None:
                continue
            for line in f:
                if not line.strip():
                    continue
                pat = json.loads(line)
                pat["relation_id"] = relation_id
                rows.append(pat)

    df = pd.DataFrame(rows)
    out = DATA_RAW / spec["raw_subdir"] / "pararel_patterns.parquet"
    df.to_parquet(out, index=False)
    return {
        "name": "ParaRel",
        "source_url": spec["url"],
        "git_commit": spec["commit"],
        "raw_archive": str(raw.relative_to(REPO_ROOT)),
        "raw_archive_sha256": sha256_of(raw),
        "snapshot": str(out.relative_to(REPO_ROOT)),
        "snapshot_sha256": sha256_of(out),
        "n_rows": len(df),
        "n_relations": int(df["relation_id"].nunique()),
    }


def existing_popqa() -> dict:
    snap = DATA_RAW / "popqa" / "popqa_test.parquet"
    if not snap.exists():
        raise RuntimeError(
            f"PopQA snapshot missing at {snap}; "
            "run scripts/freeze_popularity_bucket_definition.py first"
        )
    df = pd.read_parquet(snap)
    return {
        "name": "PopQA",
        "source_url": "akariasai/PopQA (HuggingFace)",
        "split": "test",
        "snapshot": str(snap.relative_to(REPO_ROOT)),
        "snapshot_sha256": sha256_of(snap),
        "n_rows": len(df),
    }


def main() -> None:
    FROZEN.mkdir(parents=True, exist_ok=True)
    CONFIGS.mkdir(parents=True, exist_ok=True)

    print("--- LAMA T-REx ---")
    lama = freeze_lama()
    print(f"  rows={lama['n_rows']} relations={lama['n_relations']} sha256={lama['snapshot_sha256'][:16]}...")

    print("--- CounterFact ---")
    cf = freeze_counterfact()
    print(f"  rows={cf['n_rows']} sha256={cf['snapshot_sha256'][:16]}...")

    print("--- ParaRel ---")
    pr = freeze_pararel()
    print(f"  rows={pr['n_rows']} relations={pr['n_relations']} sha256={pr['snapshot_sha256'][:16]}...")

    print("--- PopQA (existing) ---")
    pop = existing_popqa()
    print(f"  rows={pop['n_rows']} sha256={pop['snapshot_sha256'][:16]}...")

    frozen_at = now_iso()
    manifest = {
        "dataset_name": "D_fact_v001",
        "version": "v001",
        "frozen_at": frozen_at,
        "sources": [lama, cf, pr, pop],
    }
    manifest_path = FROZEN / "manifest.yaml"
    with manifest_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True)
    manifest_sha = sha256_of(manifest_path)

    hash_log_path = CONFIGS / "hash_log.json"
    hash_log = json.loads(hash_log_path.read_text()) if hash_log_path.exists() else {}
    hash_log["lama_trex_snapshot_hash"] = {
        "file": lama["snapshot"], "sha256": lama["snapshot_sha256"],
        "n_rows": lama["n_rows"], "frozen_at": frozen_at,
    }
    hash_log["counterfact_snapshot_hash"] = {
        "file": cf["snapshot"], "sha256": cf["snapshot_sha256"],
        "n_rows": cf["n_rows"], "frozen_at": frozen_at,
    }
    hash_log["pararel_snapshot_hash"] = {
        "file": pr["snapshot"], "sha256": pr["snapshot_sha256"],
        "n_rows": pr["n_rows"], "git_commit": PARAREL_COMMIT, "frozen_at": frozen_at,
    }
    hash_log["d_fact_v001_manifest_hash"] = {
        "file": str(manifest_path.relative_to(REPO_ROOT)),
        "sha256": manifest_sha, "frozen_at": frozen_at,
    }
    hash_log_path.write_text(json.dumps(hash_log, indent=2, ensure_ascii=False, sort_keys=True))

    print(f"\nmanifest    : {manifest_path}  sha256={manifest_sha}")
    print(f"hash_log    : {hash_log_path}")


if __name__ == "__main__":
    main()
