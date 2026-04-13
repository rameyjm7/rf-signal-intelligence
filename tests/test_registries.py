from pathlib import Path

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_dataset_registry_has_required_fields_and_checksums():
    repo = _repo_root()
    registry = _load_yaml(repo / "configs" / "data_registry.yaml")

    assert registry["version"] == 1
    datasets = registry["datasets"]
    assert datasets, "Dataset registry must not be empty."

    for dataset in datasets:
        assert dataset.get("id")
        assert dataset.get("name")
        assert dataset.get("format")
        assert dataset.get("sample_shape")

        if "path" in dataset:
            assert isinstance(dataset["path"], str)
        if "paths" in dataset:
            assert isinstance(dataset["paths"], dict)
            assert dataset["paths"], "Dataset path map must not be empty."

        checksum = dataset.get("checksum")
        assert isinstance(checksum, dict), f"Missing checksum metadata for {dataset['id']}"
        assert checksum.get("algorithm") == "sha256"
        assert "value" in checksum


def test_model_registry_matches_known_datasets_and_shapes_and_checksums():
    repo = _repo_root()
    dataset_registry = _load_yaml(repo / "configs" / "data_registry.yaml")
    model_registry = _load_yaml(repo / "configs" / "model_registry.yaml")

    dataset_ids = {entry["id"] for entry in dataset_registry["datasets"]}

    assert model_registry["version"] == 1
    models = model_registry["models"]
    assert models, "Model registry must not be empty."

    for model in models:
        assert model.get("id")
        assert model.get("dataset") in dataset_ids
        assert isinstance(model.get("path"), str)
        assert len(model.get("input_shape", [])) == 2
        assert len(model.get("output_shape", [])) == 2
        assert model.get("description")

        checksum = model.get("checksum")
        assert isinstance(checksum, dict), f"Missing checksum metadata for {model['id']}"
        assert checksum.get("algorithm") == "sha256"
        assert "value" in checksum
