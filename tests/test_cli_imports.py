import importlib
import sys


def test_main_module_import_does_not_load_tensorflow_when_not_preloaded():
    if "tensorflow" in sys.modules:
        # If a local plugin or user env preloaded TensorFlow, we cannot assert import side effects.
        return

    sys.modules.pop("rf_signal_intelligence.__main__", None)
    importlib.import_module("rf_signal_intelligence.__main__")

    assert "tensorflow" not in sys.modules
