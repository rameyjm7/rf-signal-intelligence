import argparse
import faulthandler
from pathlib import Path

from ml_wireless_classification.base.CommonVars import RUN_MODE, common_vars
from ml_wireless_classification.rnn_lstm_w_SNR import ModulationLSTMClassifier

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
        help="Directory for saved model artifacts. Defaults to <repo>/models.",
    )
    parser.add_argument(
        "--stats-dir",
        default=None,
        help="Directory for stats artifacts. Defaults to <repo>/stats.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    repo_root = _repo_root()

    data_path = (
        Path(args.data_path).expanduser().resolve()
        if args.data_path
        else _default_data_path(repo_root)
    )
    models_dir = (
        Path(args.models_dir).expanduser().resolve()
        if args.models_dir
        else repo_root / "models"
    )
    stats_dir = (
        Path(args.stats_dir).expanduser().resolve()
        if args.stats_dir
        else repo_root / "stats"
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"{args.model_name}.keras"
    stats_path = stats_dir / f"{args.model_name}_stats.json"

    common_vars.models_dir = str(models_dir)
    common_vars.stats_dir = str(stats_dir)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Use --data-path to provide a valid file."
        )

    print(f"Mode: {args.mode}")
    print(f"Data path: {data_path}")
    print(f"Model path: {model_path}")
    print(f"Stats path: {stats_path}")

    classifier = ModulationLSTMClassifier(
        str(data_path), str(model_path), str(stats_path)
    )
    classifier.main(RUN_MODE(args.mode))


if __name__ == "__main__":
    main()
