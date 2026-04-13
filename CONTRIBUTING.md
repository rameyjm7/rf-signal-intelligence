# Contributing

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Optional sanity check:

```bash
python -m ml_wireless_classification --help
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

## Data and Artifacts

This repository contains large datasets and model artifacts.

- Do not add new large binary files unless they are required and intentionally versioned.
- Keep generated outputs out of commits where possible.
- Prefer updating `.gitignore` for local experiment outputs.

## Notebooks

- Keep exploratory work in `archive/` when not production-ready.
- For notebooks in `notebooks/`, keep outputs minimal and document expected inputs/paths.

## Style and Scope

- Prioritize reliability and reproducibility.
- Avoid unrelated refactors in the same change.
- Update documentation whenever CLI behavior, setup, or runtime paths change.
