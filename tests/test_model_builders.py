import sys
import types


class _FakeLayer:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _FakeSequential:
    def __init__(self, layers=None):
        self.layers = list(layers or [])
        self.compiled = None

    def compile(self, **kwargs):
        self.compiled = kwargs


class _FakeAdam:
    def __init__(self, learning_rate=None, *args, **kwargs):
        self.learning_rate = learning_rate
        self.kwargs = kwargs


def _install_fake_tensorflow(monkeypatch):
    tf = types.ModuleType("tensorflow")
    keras = types.ModuleType("tensorflow.keras")
    layers = types.ModuleType("tensorflow.keras.layers")
    optimizers = types.ModuleType("tensorflow.keras.optimizers")

    keras.Sequential = _FakeSequential
    optimizers.Adam = _FakeAdam
    for name in [
        "BatchNormalization",
        "Bidirectional",
        "Conv1D",
        "Dense",
        "Dropout",
        "Input",
        "LSTM",
        "MaxPooling1D",
    ]:
        setattr(layers, name, _FakeLayer)
    keras.layers = layers
    keras.optimizers = optimizers
    tf.keras = keras

    monkeypatch.setitem(sys.modules, "tensorflow", tf)
    monkeypatch.setitem(sys.modules, "tensorflow.keras", keras)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.layers", layers)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.optimizers", optimizers)


def test_rml2018_lstm_builder_compiles_model(monkeypatch):
    _install_fake_tensorflow(monkeypatch)
    from ml_wireless_classification.models.rml2018_lstm import build_rml2018_lstm_model

    model = build_rml2018_lstm_model((1024, 3), 24, learning_rate=1e-4)

    assert isinstance(model, _FakeSequential)
    assert len(model.layers) == 7
    assert model.compiled["loss"] == "sparse_categorical_crossentropy"


def test_deepradar_cnn_bilstm_builder_compiles_model(monkeypatch):
    _install_fake_tensorflow(monkeypatch)
    from ml_wireless_classification.models.deepradar_cnn_bilstm import build_deepradar_cnn_bilstm

    model = build_deepradar_cnn_bilstm((1024, 3), 23)

    assert isinstance(model, _FakeSequential)
    assert len(model.layers) == 13
    assert model.compiled["loss"] == "categorical_crossentropy"
