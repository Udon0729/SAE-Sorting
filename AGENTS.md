# Repository Guidelines

## Project Structure & Module Organization

This repository supports SAE feature-cluster sorting experiments. Keep reusable Python package code in `src/sae_sorting/`; place runnable dataset and utility generators in `scripts/`; and keep phase-specific research workflows in `experiments/`. Configuration lives in `configs/`, preregistration notes in `docs/preregistrations/`, operational helpers in `ops/`, and versioned generated artifacts in `data/` such as `data/synthetic_kg/v001/` and `data/utility/v001/`. Use `tests/` for automated tests; it is currently empty, so new behavior should add focused coverage there.

## Build, Test, and Development Commands

- `uv sync`: create or update the local Python 3.12 environment from `pyproject.toml` and `uv.lock`.
- `uv run python scripts/generate_d_synth_v001.py --size small`: generate the small synthetic knowledge graph dataset for local checks.
- `uv run python scripts/freeze_d_fact_v001.py`: freeze derived fact data and hashes after generation.
- `uv run python experiments/smoke_pythia_160m.py`: run the lightweight smoke experiment.
- `uv run python -m py_compile scripts/*.py experiments/*.py src/sae_sorting/*.py`: catch syntax errors before committing.

## Coding Style & Naming Conventions

Use Python 3.12 with 4-space indentation, type hints where helpful, and `from __future__ import annotations` in new modules. Follow the existing script style: module docstring with inputs, outputs, and example run command; constants in `UPPER_SNAKE_CASE`; functions in `snake_case`; and `Path` objects rooted at `REPO_ROOT`. Name phase scripts and artifacts with explicit phase/version markers, for example `phase4_decide.py`, `d_fact_v001`, or `phase3_*_rho050_v2.parquet`.

## Testing Guidelines

Prefer deterministic tests for data-shape, hashing, and gate-decision logic. Add tests as `tests/test_<area>.py`, and keep fixtures small enough to run without model downloads. Until a formal pytest setup is added, run the smoke script and `py_compile` command above. When changing generation code, verify updated `hashes/sha256sums.txt` and `configs/hash_log.json` entries are intentional.

## Commit & Pull Request Guidelines

Git history uses short, imperative, phase-scoped subjects, often lowercase: `fix attribution script and decide script`, `phase4 v001 pre-registration`, or `phase4 v001 implementation ... -> FREEZE`. Keep commits focused on one phase or artifact family. Pull requests should summarize the phase/version affected, list regenerated data paths, note hash-log changes, and include the exact commands run. Link preregistration docs when changing acceptance gates or freeze decisions.

## Security & Configuration Tips

Do not commit credentials, model checkpoints, or large transient outputs unless they are documented frozen artifacts. Keep generated data versioned under `data/<dataset>/vNNN/`, and update manifests or hash logs whenever outputs become part of the reproducible record.
