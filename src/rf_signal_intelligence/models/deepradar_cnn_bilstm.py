"""DeepRadar2022 CNN+BiLSTM architecture extracted from notebook 32."""

from __future__ import annotations


def build_deepradar_cnn_bilstm(
    input_shape: tuple[int, int] = (1024, 3),
    num_classes: int = 23,
    *,
    learning_rate: float = 5e-4,
    clipnorm: float = 1.0,
):
    """Build the notebook-32 CNN + bidirectional LSTM model."""
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import (
        LSTM,
        BatchNormalization,
        Bidirectional,
        Conv1D,
        Dense,
        Dropout,
        Input,
        MaxPooling1D,
    )

    model = Sequential(
        [
            Input(shape=input_shape),
            Conv1D(64, 5, activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling1D(2),
            Conv1D(128, 3, activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling1D(2),
            Bidirectional(LSTM(128, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(64)),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(num_classes, activation="softmax", dtype="float32"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
