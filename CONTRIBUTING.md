# Contributing

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,test]"
pre-commit install
```

Optional sanity check:

```bash
python -m ml_wireless_classification --help
```

Lint/format checks:

```bash
ruff check src tests
ruff format src tests
pytest -q -m "not integration"
```

Optional integration checks:

```bash
python -m pytest -q -m integration -rs
```

## Branching

- Create feature branches from `main` (or the active integration branch when coordinated).
- Keep changes scoped and atomic.
- Use descriptive branch names, for example: `feature/cli-cleanup`, `fix/dockerfile-syntax`.

## Commits

- Use clear, imperative commit messages.
- Prefer one logical change per commit.
- Include relevant context in the body when behavior or interfaces change.

## Pull Requests

- Summarize what changed and why.
- Call out runtime impacts (training behavior, data paths, model artifacts, Docker behavior).
- Include reproducible commands for validation.
- Keep pre-commit clean before pushing:
  - `pre-commit run --all-files`

## Data and Artifacts

This repository contains large datasets and model artifacts.

- Do not add new large binary files unless they are required and intentionally versioned.
- Keep generated outputs out of commits where possible.
- Prefer updating `.gitignore` for local experiment outputs.
- Runtime outputs belong under `outputs/` unless a task explicitly requires another path.

## Registries

- Keep dataset metadata in `configs/data_registry.yaml`.
- Keep model metadata in `configs/model_registry.yaml` (including input/output shapes).
- Keep checksum metadata (`checksum.algorithm`, `checksum.value`) present for each registry entry.
- Use `python scripts/update_registry_checksums.py` to refresh checksum values.

## Notebooks

- Keep exploratory work in `archive/` when not production-ready.
- For notebooks in `notebooks/`, keep outputs minimal and document expected inputs/paths.

## Style and Scope

- Prioritize reliability and reproducibility.
- Avoid unrelated refactors in the same change.
- Update documentation whenever CLI behavior, setup, or runtime paths change.

## Release Policy

- Follow `RELEASE.md` for version bumps, tagging, and release notes.
- Update `CHANGELOG.md` in every PR that changes behavior, interfaces, or outputs.

## Source Layout

- Put maintained runtime code in `src/ml_wireless_classification/core/`.
- Put maintained model definitions in `src/ml_wireless_classification/models/`.
- Keep older experimental helpers in `src/ml_wireless_classification/legacy/`.
- Do not add new logic to `src/ml_wireless_classification/base/`; it exists for backward-compatible wrappers.
