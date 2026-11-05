import json
from pathlib import Path
import tensorflow as tf
from unet_model import unet
from config import config


# ---------------------------------------------------------------------------
# Metric helpers — named functions (not lambdas) to avoid Keras serialization
# warnings when saving in .keras format.
# ---------------------------------------------------------------------------

def _metric_psnr(y_true, y_pred):
    """Per-sample PSNR in [0,1] space; used as a Keras metric."""
    return tf.image.psnr(y_pred, y_true, max_val=1.0)


def _metric_ssim(y_true, y_pred):
    """Per-sample SSIM in [0,1] space; used as a Keras metric."""
    return tf.image.ssim(y_pred, y_true, max_val=1.0)


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_unet(
    train_data,
    val_data=None,
    y_train=None,
    y_val=None,
    input_shape=config.input_shape,
    output_dir=config.unet_model_dir,
    num_epochs=config.unet_epochs,
    batch_size=config.unet_batch_size,
    model_name="my_Model",
    preprocess_mode=config.preprocess_mode,
):
    """Train the U-Net dehazing model.

    Supports both tf.data.Dataset inputs (Phase 3) and legacy numpy arrays:
    - tf.data call: train_unet(train_dataset, val_dataset, output_dir=..., ...)
    - legacy call:  train_unet(X_train, X_val, y_train, y_val, input_shape, output_dir, ...)

    Args:
        train_data: tf.data.Dataset yielding (hazy, gt) pairs, or X_train array.
        val_data:   tf.data.Dataset yielding (hazy, gt) pairs, or X_val array.
        y_train:    Optional ground truth array for legacy calling convention.
        y_val:      Optional ground truth array for legacy calling convention.
        input_shape: Input shape tuple (H, W, C).
        output_dir: Directory where the model and sidecar JSON are saved.
        num_epochs: Number of training epochs.
        batch_size: Mini-batch size.
        model_name: Base filename (no extension).
        preprocess_mode: Preprocessing mode ("clahe_hls" | "clahe" | "none").
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = unet(input_shape)

    # CG3: use MSE loss (pixel-space, correct for normalized [0,1] images)
    # CG4: add PSNR and SSIM as named-function metrics for live validation logging
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanSquaredError(name="mse"),
            tf.keras.metrics.MeanMetricWrapper(fn=_metric_psnr, name="psnr"),
            tf.keras.metrics.MeanMetricWrapper(fn=_metric_ssim, name="ssim"),
        ],
    )

    if isinstance(train_data, tf.data.Dataset):
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=num_epochs,
        )
    elif y_train is not None and y_val is not None:
        history = model.fit(
            train_data, y_train,
            validation_data=(val_data, y_val),
            epochs=num_epochs,
            batch_size=batch_size,
        )
    else:
        # Fallback: assume train_data has fit interface
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=num_epochs,
        )

    # CG4: save in .keras format (primary) + .h5 for compatibility
    keras_path = output_dir / f"{model_name}.keras"
    h5_path = output_dir / f"{model_name}.h5"
    model.save(str(keras_path))
    model.save(str(h5_path))
    print(f"[Save] U-Net saved to {keras_path} and {h5_path}")

    # CG3: save preprocess mode sidecar JSON
    sidecar_meta = {
        "model_name": model_name,
        "model_type": "unet",
        "preprocess_mode": preprocess_mode,
        "input_shape": list(input_shape),
    }
    sidecar_path = output_dir / f"{model_name}.json"
    sidecar_path.write_text(json.dumps(sidecar_meta, indent=2), encoding="utf-8")
    print(f"[Save] Preprocess mode sidecar saved to {sidecar_path}")

    return model, history
