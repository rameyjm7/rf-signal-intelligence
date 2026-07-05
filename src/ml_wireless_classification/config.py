"""Config loading helpers for reproducible RFML workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    """Return the repository root when running from an installed or editable package."""
    return Path(__file__).resolve().parents[2]


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return an empty dict for empty files."""
    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping at {config_path}, got {type(data).__name__}")
    return data


def resolve_path(value: str | Path, *, base_dir: str | Path | None = None) -> Path:
    """Resolve a config path relative to a base directory unless it is absolute."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    base = Path(base_dir).expanduser() if base_dir is not None else repo_root()
    return (base / path).resolve()


@dataclass(frozen=True)
class WorkflowPaths:
    """Common project paths used by CLI workflows."""

    project_root: Path
    outputs_dir: Path
    models_dir: Path

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> WorkflowPaths:
        root = resolve_path(config.get("project_root", repo_root()))
        outputs = resolve_path(config.get("outputs_dir", "outputs"), base_dir=root)
        models = resolve_path(config.get("models_dir", "models"), base_dir=root)
        return cls(project_root=root, outputs_dir=outputs, models_dir=models)
