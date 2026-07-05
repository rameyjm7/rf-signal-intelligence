"""Modern `rfsi` command-line interface for reusable workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml_wireless_classification.config import load_yaml_config, resolve_path
from ml_wireless_classification.workflows.comparison import (
    build_comparison_rows,
    write_comparison_outputs,
)
from ml_wireless_classification.workflows.noisy_drone_vgg import (
    evaluate_noisy_drone_vgg,
    export_noisy_drone_vgg_to_onnx,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rfsi",
        description="Reusable RF Signal Intelligence workflows.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    train = subcommands.add_parser("train", help="Train a configured model workflow.")
    train.add_argument("--config", required=True, help="Workflow YAML config.")

    evaluate = subcommands.add_parser("evaluate", help="Evaluate a configured model workflow.")
    evaluate.add_argument("--config", required=True, help="Workflow YAML config.")
    evaluate.add_argument("--checkpoint", help="Override model checkpoint path.")

    compare = subcommands.add_parser("compare", help="Reproduce a cross-dataset comparison table.")
    compare.add_argument("--config", required=True, help="Comparison YAML config.")

    export = subcommands.add_parser("export-onnx", help="Export a configured model to ONNX.")
    export.add_argument("--config", required=True, help="Workflow YAML config.")
    export.add_argument("--checkpoint", help="Override model checkpoint path.")
    export.add_argument("--out", help="Override ONNX output path.")
    return parser


def _load_workflow_config(path: str | Path) -> dict:
    config = load_yaml_config(path)
    config.setdefault("project_root", str(Path(path).resolve().parents[1]))
    return config


def run_train(args: argparse.Namespace) -> int:
    config = _load_workflow_config(args.config)
    workflow = config.get("workflow")
    if workflow == "noisy_drone_vgg":
        raise NotImplementedError(
            "Config-driven NoisyDroneRFv2 training is planned; use notebook 33 until "
            "the training loop is fully migrated. Evaluation/export are available now."
        )
    raise ValueError(f"Unsupported training workflow: {workflow!r}")


def run_evaluate(args: argparse.Namespace) -> int:
    config = _load_workflow_config(args.config)
    if args.checkpoint:
        config.setdefault("model", {})["checkpoint"] = args.checkpoint
    workflow = config.get("workflow")
    if workflow == "noisy_drone_vgg":
        metrics = evaluate_noisy_drone_vgg(config)
        print(metrics)
        return 0
    raise ValueError(f"Unsupported evaluation workflow: {workflow!r}")


def run_compare(args: argparse.Namespace) -> int:
    config = _load_workflow_config(args.config)
    project_root = resolve_path(config.get("project_root", "."))
    rows = build_comparison_rows(config, project_root=project_root)
    csv_path, json_path = write_comparison_outputs(
        rows,
        output_csv=resolve_path(config["output_csv"], base_dir=project_root),
        output_json=resolve_path(config["output_json"], base_dir=project_root),
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    return 0


def run_export_onnx(args: argparse.Namespace) -> int:
    config = _load_workflow_config(args.config)
    if args.checkpoint:
        config.setdefault("model", {})["checkpoint"] = args.checkpoint
    if args.out:
        config.setdefault("export", {})["onnx_path"] = args.out
    workflow = config.get("workflow")
    if workflow == "noisy_drone_vgg":
        onnx_path = export_noisy_drone_vgg_to_onnx(config)
        print(f"Wrote {onnx_path}")
        return 0
    raise ValueError(f"Unsupported ONNX export workflow: {workflow!r}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        return run_train(args)
    if args.command == "evaluate":
        return run_evaluate(args)
    if args.command == "compare":
        return run_compare(args)
    if args.command == "export-onnx":
        return run_export_onnx(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
