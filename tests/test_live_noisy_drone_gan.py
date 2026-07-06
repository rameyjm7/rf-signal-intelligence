from __future__ import annotations

import numpy as np

from rf_signal_intelligence.live_noisy_drone_rf_classifier import (
    generate_gan_iq,
    infer_generator_latent_dim,
)


class FakeGenerator:
    input_shape = [(None, 5), (None,)]

    def __init__(self) -> None:
        self.calls = []

    def predict(self, inputs, *, batch_size: int, verbose: int):
        self.calls.append((inputs, batch_size, verbose))
        labels = np.asarray(inputs["label"], dtype=np.float32)
        out = np.zeros((len(labels), 4, 2), dtype=np.float32)
        out[:, :, 0] = labels[:, None]
        out[:, :, 1] = np.arange(4, dtype=np.float32)[None, :]
        return out


def test_infer_generator_latent_dim_uses_noise_input_shape():
    assert infer_generator_latent_dim(FakeGenerator()) == 5


def test_generate_gan_iq_concatenates_class_conditioned_windows():
    generator = FakeGenerator()

    iq = generate_gan_iq(generator, class_idx=3, seed=123, samples=2, batch_size=7)

    assert iq.shape == (8, 2)
    assert iq.dtype == np.float32
    assert np.all(iq[:, 0] == 3.0)
    assert np.array_equal(iq[:4, 1], np.arange(4, dtype=np.float32))
    assert generator.calls[0][1:] == (7, 0)
