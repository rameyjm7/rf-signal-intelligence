"""RML2016 LSTM architecture builders extracted from notebook 30."""

from __future__ import annotations


def build_rml2016_lstm_model(
    input_shape: tuple[int, int] = (128, 3),
    num_classes: int = 11,
    *,
    learning_rate: float = 1e-4,
):
    """Build the notebook-30 RML2016 LSTM model."""
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    model = Sequential(
        [
            LSTM(256, input_shape=input_shape, return_sequences=True),
            Dropout(0.5),
            LSTM(256, return_sequences=False),
            Dropout(0.4),
            Dense(512, activation="relu"),
            Dropout(0.5),
            Dense(256, activation="relu"),
            Dropout(0.4),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(96, activation="relu"),
            Dropout(0.25),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dropout(0.1),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
