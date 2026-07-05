"""RML2018 LSTM architecture builders extracted from notebook 31."""

from __future__ import annotations


def build_rml2018_lstm_model(
    input_shape: tuple[int, int] = (1024, 3),
    num_classes: int = 24,
    *,
    learning_rate: float = 1e-4,
):
    """Build the notebook-31 RML2018 LSTM model."""
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    model = Sequential(
        [
            LSTM(128, input_shape=input_shape, return_sequences=True),
            Dropout(0.5),
            LSTM(128, return_sequences=False),
            Dropout(0.2),
            Dense(128, activation="relu"),
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


def auto_select_batch_size(model, x_train, y_train, candidates, fallback: int = 64) -> int:
    """Select the largest batch size that fits GPU memory using a one-step dry run."""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print(f"No GPU detected. Using fallback batch size: {fallback}")
        return int(fallback)

    selected = None
    for batch_size in sorted(set(int(value) for value in candidates), reverse=True):
        if batch_size <= 0 or len(x_train) < batch_size:
            continue
        try:
            probe = tf.keras.models.clone_model(model)
            probe.build(model.input_shape)
            probe.set_weights(model.get_weights())
            optimizer = type(model.optimizer).from_config(model.optimizer.get_config())
            probe.compile(optimizer=optimizer, loss=model.loss, metrics=["accuracy"])
            probe.train_on_batch(x_train[:batch_size], y_train[:batch_size])
            selected = batch_size
            print(f"Batch size {batch_size} fits GPU memory.")
            break
        except tf.errors.ResourceExhaustedError:
            print(f"Batch size {batch_size} OOM; trying smaller size...")
        except Exception as exc:
            print(f"Batch size {batch_size} probe failed ({type(exc).__name__}); trying smaller size...")

    if selected is None:
        selected = int(fallback)
        print(f"No candidate batch size succeeded. Using fallback: {selected}")
    else:
        print(f"Selected batch size: {selected}")
    return selected
