from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEPLOYMENT_FILES = [
    "exports/export_noisy_drone_to_onnx.py",
    "exports/validate_onnx.py",
    "exports/run_onnx_inference.py",
    "deploy/build_tensorrt_engine.sh",
    "deploy/run_trtexec_benchmark.sh",
    "deploy/run_jetson_inference.py",
    "deploy/nsight_profile.sh",
    "docs/jetson_tensorrt_deployment.md",
]


def test_deployment_artifacts_exist():
    for relative_path in DEPLOYMENT_FILES:
        assert (REPO_ROOT / relative_path).exists(), f"Missing deployment artifact: {relative_path}"


def test_jetson_deployment_doc_covers_required_pipeline_steps():
    text = (REPO_ROOT / "docs/jetson_tensorrt_deployment.md").read_text(encoding="utf-8")
    for phrase in [
        "Keras model",
        "ONNX export",
        "ONNX validation",
        "TensorRT FP16 engine",
        "Jetson inference",
        "trtexec benchmark",
        "Nsight Systems profile",
    ]:
        assert phrase in text


def test_export_scripts_expose_expected_flags():
    export_text = (REPO_ROOT / "exports/export_noisy_drone_to_onnx.py").read_text(encoding="utf-8")
    validate_text = (REPO_ROOT / "exports/validate_onnx.py").read_text(encoding="utf-8")

    for flag in ["--keras-model", "--out", "--sample-out", "--labels-out"]:
        assert flag in export_text
    for flag in ["--keras-model", "--onnx", "--sample", "--labels"]:
        assert flag in validate_text


def test_local_onnx_inference_script_exposes_expected_flags():
    text = (REPO_ROOT / "exports/run_onnx_inference.py").read_text(encoding="utf-8")

    for flag in [
        "--onnx",
        "--input",
        "--iq-file",
        "--labels",
        "--providers",
        "--top-k",
        "--decision-mode",
        "--target-class",
        "--window-score-mode",
        "--class-sweep",
        "--dataset-dir",
        "--samples-per-class",
        "--max-predictions",
        "--min-snr",
        "--format",
    ]:
        assert flag in text


def test_gateway_rx_classifier_exposes_jetson_event_flags():
    text = (REPO_ROOT / "scripts/noisy_drone_gateway_rx_classifier.py").read_text(encoding="utf-8")

    for flag in [
        "--backend",
        "--engine",
        "--continuous",
        "--event-jsonl",
        "--run-id",
        "event-json",
        "end_to_end",
        "preprocess",
        "inference",
    ]:
        assert flag in text
