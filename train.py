import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")  # save plots to files, no pop-up windows
import matplotlib.pyplot as plt

import config

tf.keras.utils.set_random_seed(config.SEED)


def check_folders():
    for folder in (config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR):
        if not os.path.isdir(folder):
            raise SystemExit(
                f"Folder not found: {folder}\n"
                "Place your dataset as dataset/train, dataset/validation, dataset/test. "
                "If you have no validation folder, run: python prepare_validation.py"
            )


def load_datasets():
    common = dict(
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR, shuffle=True, seed=config.SEED, **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        config.VAL_DIR, shuffle=False, **common)

    class_names = train_ds.class_names
    if val_ds.class_names != class_names:
        raise SystemExit(
            f"Class folders differ!\ntrain: {class_names}\nvalidation: {val_ds.class_names}")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    return train_ds, val_ds, class_names


def get_class_weights(class_names):
    """Give rare classes a larger weight so the model does not ignore them."""
    labels = []
    for index, name in enumerate(class_names):
        folder = os.path.join(config.TRAIN_DIR, name)
        count = len([f for f in os.listdir(folder)
                     if f.lower().endswith(config.IMAGE_EXTENSIONS)])
        labels.extend([index] * count)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(class_names)),
        y=np.array(labels),
    )
    return {i: float(w) for i, w in enumerate(weights)}


def build_model(num_classes):
    # Augmentation: only active during training, automatically off at prediction.
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")

    base_model = MobileNetV2(
        input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # phase 1: frozen

    inputs = layers.Input(shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3), name="input_image")
    x = data_augmentation(inputs)
    # MobileNetV2 expects pixels in [-1, 1]; this layer converts 0-255 to that range.
    x = layers.Rescaling(1.0 / 127.5, offset=-1, name="rescaling")(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="alzheimer_mobilenetv2")
    return model, base_model


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )


def make_callbacks(initial_best=None):
    checkpoint_args = dict(
        filepath=config.MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )
    if initial_best is not None:
        # Phase 2 only overwrites the saved model if it beats phase 1.
        checkpoint_args["initial_value_threshold"] = initial_best
    return [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        ModelCheckpoint(**checkpoint_args),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7, verbose=1),
    ]


def plot_history(history):
    epochs = range(1, len(history["accuracy"]) + 1)
    split = config.EPOCHS_FROZEN

    for metric, title, filename in (
        ("accuracy", "Training and Validation Accuracy", "accuracy_graph.png"),
        ("loss", "Training and Validation Loss", "loss_graph.png"),
    ):
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history[metric], label=f"Training {metric}")
        plt.plot(epochs, history[f"val_{metric}"], label=f"Validation {metric}")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(config.OUTPUT_DIR, filename), dpi=150)
        plt.close()


def main():
    check_folders()
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    train_ds, val_ds, class_names = load_datasets()
    num_classes = len(class_names)
    print("Classes found:", class_names)

    # Save class order so prediction uses the exact same mapping.
    with open(config.CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f, indent=2)

    class_weights = get_class_weights(class_names)
    print("Class weights:", class_weights)

    model, base_model = build_model(num_classes)

    # ---------- Phase 1: train only the new head ----------
    print("\n=== Phase 1: training classifier head (base frozen) ===")
    compile_model(model, config.LEARNING_RATE_FROZEN)
    h1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=config.EPOCHS_FROZEN,
        class_weight=class_weights,
        callbacks=make_callbacks(),
    )
    best_val_loss = min(h1.history["val_loss"])

    # ---------- Phase 2: fine-tune the last layers ----------
    print("\n=== Phase 2: fine-tuning last layers of MobileNetV2 ===")
    base_model.trainable = True
    for layer in base_model.layers[:-config.FINE_TUNE_LAYERS]:
        layer.trainable = False
    for layer in base_model.layers:  # keep BatchNorm frozen for stable fine-tuning
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    compile_model(model, config.LEARNING_RATE_FINE_TUNE)  # recompile after changing trainable
    h2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=config.EPOCHS_FINE_TUNE,
        class_weight=class_weights,
        callbacks=make_callbacks(initial_best=best_val_loss),
    )

    history = {k: [float(v) for v in h1.history[k] + h2.history[k]] for k in h1.history}
    with open(os.path.join(config.OUTPUT_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    plot_history(history)

    print("\nTraining finished.")
    print("Best model saved to:", config.MODEL_PATH)
    print("Graphs saved in:", config.OUTPUT_DIR)
    print("Next step: python evaluate.py")


if __name__ == "__main__":
    main()