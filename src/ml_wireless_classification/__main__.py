import argparse
import faulthandler
from pathlib import Path

from ml_wireless_classification.core.run_mode import RUN_MODE, common_vars

faulthandler.enable()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_data_path(repo_root: Path) -> Path:
    candidates = [
        repo_root / "data" / "RML2016" / "RML2016.10a_dict.pkl",
        repo_root / "RML2016.10a_dict.pkl",
        Path("/workspace/code/data/RML2016/RML2016.10a_dict.pkl"),
        Path("/workspace/code/RML2016.10a_dict.pkl"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _default_outputs_root(repo_root: Path) -> Path:
    return repo_root / "outputs"


def _resolve_models_dir(repo_root: Path, model_name: str, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()

    legacy_models_dir = repo_root / "models"
    legacy_model_path = legacy_models_dir / f"{model_name}.keras"
    if legacy_model_path.exists():
        return legacy_models_dir

    return _default_outputs_root(repo_root) / "models"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or evaluate the wireless modulation RNN-LSTM model."
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in RUN_MODE],
        default=RUN_MODE.EVALUATE_ONLY.value,
        help="Execution mode.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to RML2016.10a_dict.pkl. Defaults to common repo/container locations.",
    )
    parser.add_argument(
        "--model-name",
        default="rnn_lstm_w_SNR",
        help="Base model name used for saved model and stats files.",
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help=(
            "Directory for model artifacts. Defaults to <repo>/models if an existing model "
            "matches --model-name, otherwise <repo>/outputs/models."
        ),
    )
    parser.add_argument(
        "--stats-dir",
        default=None,
        help="Directory for stats artifacts. Defaults to <repo>/outputs/stats.",
    )
    parser.add_argument(
        "--outputs-dir",
        default=None,
        help="Root directory for generated outputs. Defaults to <repo>/outputs.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    repo_root = _repo_root()
    outputs_root = (
        Path(args.outputs_dir).expanduser().resolve()
        if args.outputs_dir
        else _default_outputs_root(repo_root)
    )

    data_path = (
        Path(args.data_path).expanduser().resolve()
        if args.data_path
        else _default_data_path(repo_root)
    )
    models_dir = _resolve_models_dir(repo_root, args.model_name, args.models_dir)
    stats_dir = (
        Path(args.stats_dir).expanduser().resolve()
        if args.stats_dir
        else outputs_root / "stats"
    )
    logs_dir = outputs_root / "logs"

    models_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"{args.model_name}.keras"
    stats_path = stats_dir / f"{args.model_name}_stats.json"

    common_vars.outputs_dir = str(outputs_root)
    common_vars.models_dir = str(models_dir)
    common_vars.stats_dir = str(stats_dir)
    common_vars.logs_dir = str(logs_dir)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Use --data-path to provide a valid file."
        )

    print(f"Mode: {args.mode}")
    print(f"Data path: {data_path}")
    print(f"Model path: {model_path}")
    print(f"Stats path: {stats_path}")

    # Delay model import until runtime; keeps `--help` and simple CLI actions lightweight.
    from ml_wireless_classification.models.rnn_lstm_with_snr import ModulationLSTMClassifier

    classifier = ModulationLSTMClassifier(
        str(data_path), str(model_path), str(stats_path)
    )
    classifier.main(RUN_MODE(args.mode))


if __name__ == "__main__":
    main()
