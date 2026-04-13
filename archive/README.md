# Archive Directory

This directory stores historical and exploratory artifacts that are not part of the primary reproducible pipeline.

## Structure

- `notebooks/legacy/`: older standalone notebooks moved from the former `scratch/` root
- `notebooks/testing/`: general experimental/testing notebooks
- `notebooks/open_set_incremental_learning/`: open-set and incremental-learning experiments
- `notebooks/gan/`: GAN-related notebook experiments
- `scripts/gan/`: archived GAN utility scripts

## Usage Guidance

- Do not treat archive content as production or canonical training/evaluation code.
- Prefer adding maintained workflows to `src/`, `tests/`, and `notebooks/`.
- Keep archive files read-only unless actively restoring or migrating an experiment.
