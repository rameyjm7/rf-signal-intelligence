# Changelog

All notable changes to this project are documented in this file.

The format follows Keep a Changelog and semantic versioning conventions.

## [Unreleased]

### Added
- `pyproject.toml` packaging config with optional dependency groups (`dev`, `test`, `gpu`).
- GitHub Actions jobs for lint/unit/integration plus published pytest JUnit reports.
- `configs/data_registry.yaml` and `configs/model_registry.yaml` with model/data metadata and checksum fields.
- Registry validation tests and integration artifact policy controls.
- Registry-driven smoke evaluation script (`scripts/smoke_eval_registry.py`).
- Pre-commit configuration for Ruff and notebook output stripping.

### Changed
- Default runtime output layout to `outputs/` for stats/logs/new artifacts.
- Standardized base compatibility module filenames to snake_case.

## [0.1.0] - 2026-04-13

### Added
- Professionalized repository structure, README, and contribution guidance.
- Multi-dataset integration tests for high-SNR evaluation.
