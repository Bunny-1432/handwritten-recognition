"""
train_and_export_tfjs.py
========================
Train a CNN on MNIST using Keras and export directly to TF.js format.

This script:
  1. Loads MNIST from keras.datasets
  2. Builds a CNN matching the UI architecture (Conv→BN→ReLU blocks)
  3. Trains for ~10 epochs (target: >99% test accuracy)
  4. Exports to TF.js LayersModel format → ../web/data/model/

Usage:
    python train_and_export_tfjs.py

Requirements:
    pip install tensorflow tensorflowjs
"""

import os
import json
import numpy as np

# ─── Suppress TF verbose logs ───────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflowjs as tfjs


# ────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────
BATCH_SIZE   = 128
EPOCHS       = 12
LR           = 1e-3
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "web", "data", "model")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "web", "data", "training_metrics.json")


# ────────────────────────────────────────────────────────────────────────────
# Data
# ────────────────────────────────────────────────────────────────────────────
def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize to [0, 1] and add channel dim → (N, 28, 28, 1)
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0
    x_train = x_train[..., np.newaxis]
    x_test  = x_test[..., np.newaxis]

    # One-hot encode
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test  = keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


# ────────────────────────────────────────────────────────────────────────────
# Model  (matches the architecture shown in the web UI)
# ────────────────────────────────────────────────────────────────────────────
def build_cnn(num_classes: int = 10) -> keras.Model:
    """
    Architecture (mirrors HandwrittenCNN in model.py):
        Input 1×28×28
        ↓ Conv(32, 3×3) + BN + ReLU + MaxPool(2)   → 32×14×14
        ↓ Conv(64, 3×3) + BN + ReLU + MaxPool(2)   → 64×7×7
        ↓ Conv(128, 3×3) + BN + ReLU               → 128×7×7
        ↓ GlobalAvgPool                             → 128
        ↓ Dense(256) + ReLU + Dropout(0.5)
        ↓ Dense(num_classes, softmax)
    """
    inp = keras.Input(shape=(28, 28, 1), name="input")

    # Block 1
    x = layers.Conv2D(32, 3, padding="same", use_bias=False, name="conv1")(inp)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling2D(2, name="pool1")(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same", use_bias=False, name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling2D(2, name="pool2")(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same", use_bias=False, name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)

    # Classifier
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    out = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inp, out, name="HandwrittenCNN_MNIST")
    return model


# ────────────────────────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────────────────────────
def train():
    print("=" * 60)
    print("NeuroScribe — MNIST CNN Training + TF.js Export")
    print("=" * 60)

    # Load data
    (x_train, y_train), (x_test, y_test) = load_mnist()
    print(f"Train: {x_train.shape}  Test: {x_test.shape}")

    # Build model
    model = build_cnn(num_classes=10)
    model.summary()

    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Callbacks
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1
        ),
    ]

    # Train
    import time
    t0 = time.time()
    history = model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)

    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n✓ Test accuracy: {test_acc*100:.2f}%  |  Test loss: {test_loss:.4f}")
    print(f"✓ Training time: {mins}m {secs}s")

    return model, history, test_acc, test_loss, total_params, f"{mins}m {secs}s"


# ────────────────────────────────────────────────────────────────────────────
# Export to TF.js
# ────────────────────────────────────────────────────────────────────────────
def export_to_tfjs(model: keras.Model, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    tfjs.converters.save_keras_model(model, output_dir)
    print(f"\n✓ TF.js model exported to: {output_dir}")
    # List exported files
    for f in os.listdir(output_dir):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"  {f:40s}  {size/1024:.1f} KB")


# ────────────────────────────────────────────────────────────────────────────
# Update training_metrics.json
# ────────────────────────────────────────────────────────────────────────────
def update_metrics(history, test_acc: float, test_loss: float,
                   total_params: int, training_time: str,
                   metrics_path: str):
    # Load existing metrics if present
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        metrics = {}

    train_acc   = [float(v) for v in history.history["accuracy"]]
    val_acc     = [float(v) for v in history.history["val_accuracy"]]
    train_loss  = [float(v) for v in history.history["loss"]]
    val_loss    = [float(v) for v in history.history["val_loss"]]
    n_epochs    = len(train_acc)

    metrics["mnist"] = {
        "test_accuracy":  round(test_acc * 100, 2),
        "test_loss":      round(test_loss, 4),
        "epochs":         n_epochs,
        "total_params":   total_params,
        "training_time":  training_time,
        "train_accuracy": train_acc,
        "val_accuracy":   val_acc,
        "train_loss":     train_loss,
        "val_loss":       val_loss,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✓ Metrics updated: {metrics_path}")


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, history, test_acc, test_loss, total_params, training_time = train()
    export_to_tfjs(model, OUTPUT_DIR)
    update_metrics(history, test_acc, test_loss, total_params, training_time, METRICS_PATH)

    print("\n" + "=" * 60)
    print(f"  Final Test Accuracy : {test_acc*100:.2f}%")
    print(f"  Model exported to   : {OUTPUT_DIR}")
    print("=" * 60)
