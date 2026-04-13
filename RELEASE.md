# Release Process

1. Update `CHANGELOG.md` under `## [Unreleased]`.
2. Ensure local quality checks pass:
   - `ruff check src tests`
   - `python -m pytest -q -m "not integration"`
   - `python -m pytest -q -m integration -rs` (if artifacts are available)
3. Bump `project.version` in `pyproject.toml`.
4. Create a release commit: `git commit -m "release: vX.Y.Z"`.
5. Tag the release: `git tag vX.Y.Z`.
6. Push branch and tags: `git push origin <branch> --tags`.
7. Create GitHub Release notes from the tagged changelog section.
