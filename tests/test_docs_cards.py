from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_CARDS = [
    "docs/model_cards/noisy_drone_rf_v2_vgg.md",
    "docs/model_cards/rml2016_cnn_transformer.md",
    "docs/model_cards/rml2018_lstm.md",
    "docs/model_cards/deepradar2022_cnn_transformer.md",
]

DATASET_CARDS = [
    "docs/dataset_cards/noisy_drone_rf_v2.md",
    "docs/dataset_cards/rml2016.md",
    "docs/dataset_cards/rml2018.md",
    "docs/dataset_cards/deepradar2022.md",
]

MODEL_SECTIONS = [
    "## Intended Use",
    "## Input Shape",
    "## Preprocessing",
    "## Classes",
    "## Training Split",
    "## Evaluation Protocol",
    "## Headline Metrics",
    "## Known Limitations",
    "## Latency Target",
    "## Export / Deployment Status",
]

DATASET_SECTIONS = [
    "## Source",
    "## Class List",
    "## SNR Range",
    "## Sample Count",
    "## Preprocessing Assumptions",
    "## License / Usage Limitations",
    "## Leakage Risks",
    "## Recommended Evaluation Split",
]


def test_model_cards_exist_and_cover_required_sections():
    for relative_path in MODEL_CARDS:
        path = REPO_ROOT / relative_path
        assert path.exists(), f"Missing model card: {relative_path}"
        text = path.read_text(encoding="utf-8")
        for section in MODEL_SECTIONS:
            assert section in text, f"{relative_path} missing {section}"


def test_dataset_cards_exist_and_cover_required_sections():
    for relative_path in DATASET_CARDS:
        path = REPO_ROOT / relative_path
        assert path.exists(), f"Missing dataset card: {relative_path}"
        text = path.read_text(encoding="utf-8")
        for section in DATASET_SECTIONS:
            assert section in text, f"{relative_path} missing {section}"
