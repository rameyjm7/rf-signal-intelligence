from pathlib import Path

from rf_signal_intelligence.__main__ import _default_outputs_root, _resolve_models_dir


def test_default_outputs_root_uses_repo_outputs_dir(tmp_path: Path):
    assert _default_outputs_root(tmp_path) == tmp_path / "outputs"


def test_models_dir_prefers_legacy_model_if_present(tmp_path: Path):
    legacy_dir = tmp_path / "models"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "baseline.keras").touch()

    resolved = _resolve_models_dir(tmp_path, "baseline", explicit=None)
    assert resolved == legacy_dir


def test_models_dir_defaults_to_outputs_when_no_legacy_model(tmp_path: Path):
    resolved = _resolve_models_dir(tmp_path, "missing", explicit=None)
    assert resolved == tmp_path / "outputs" / "models"


def test_models_dir_honors_explicit_override(tmp_path: Path):
    explicit = tmp_path / "custom_models"
    resolved = _resolve_models_dir(tmp_path, "anything", explicit=str(explicit))
    assert resolved == explicit.resolve()
