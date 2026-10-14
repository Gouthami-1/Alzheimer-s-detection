import os
import json
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

import config

_model = None
_class_names = None


def load_resources():
    """Load the model and class names once. Safe to call many times."""
    global _model, _class_names
    if _model is not None and _class_names is not None:
        return

    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            "Trained model not found. Please run 'python train.py' first.")
    if not os.path.exists(config.CLASS_NAMES_PATH):
        raise FileNotFoundError(
            "class_names.json not found. Please run 'python train.py' first.")

    with open(config.CLASS_NAMES_PATH) as f:
        names = json.load(f)

    model = tf.keras.models.load_model(config.MODEL_PATH, compile=False)
    if model.output_shape[-1] != len(names):
        raise ValueError("Model output size does not match class_names.json. Retrain the model.")

    _model = model
    _class_names = names


def preprocess_image(image_path):
    """Same steps as training: RGB, resize to IMG_SIZE (bilinear), pixel values 0-255.
    The 0-255 -> [-1, 1] scaling happens inside the model (Rescaling layer)."""
    with Image.open(image_path) as img:
        array = np.array(img.convert("RGB"), dtype="float32")
    resized = tf.image.resize(array, config.IMG_SIZE, method="bilinear").numpy()
    batch = np.expand_dims(resized, axis=0)
    rgb_uint8 = np.clip(resized, 0, 255).astype("uint8")
    return batch, rgb_uint8


def predict_image(image_path):
    load_resources()
    batch, _ = preprocess_image(image_path)
    probabilities = _model(batch, training=False).numpy()[0]
    index = int(np.argmax(probabilities))

    all_probs = [
        {"class": config.pretty_name(name), "percent": round(float(p) * 100, 2)}
        for name, p in zip(_class_names, probabilities)
    ]
    all_probs.sort(key=lambda item: item["percent"], reverse=True)

    return {
        "class_index": index,
        "predicted_class": config.pretty_name(_class_names[index]),
        "confidence": round(float(probabilities[index]) * 100, 2),
        "probabilities": all_probs,
    }


def generate_gradcam(image_path, output_path, class_index):
    """Grad-CAM: shows which image regions pushed the model toward class_index."""
    load_resources()
    batch, rgb_uint8 = preprocess_image(image_path)

    layer_list = _model.layers
    # The MobileNetV2 base is the nested model that is not the Sequential augmentation block.
    base_model = next(l for l in layer_list
                      if isinstance(l, tf.keras.Model) and not isinstance(l, tf.keras.Sequential))
    base_index = layer_list.index(base_model)
    pre_layers = layer_list[1:base_index]   # augmentation (inactive) + rescaling
    head_layers = layer_list[base_index + 1:]  # pooling, dropout, dense

    x = tf.convert_to_tensor(batch)
    for layer in pre_layers:
        x = layer(x, training=False)

    with tf.GradientTape() as tape:
        conv_output = base_model(x, training=False)  # last convolution feature map
        tape.watch(conv_output)
        h = conv_output
        for layer in head_layers:
            h = layer(h, training=False)
        class_score = h[:, class_index]

    grads = tape.gradient(class_score, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.squeeze(conv_output[0] @ pooled_grads[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0)
    heatmap = (heatmap / (tf.reduce_max(heatmap) + 1e-8)).numpy()

    heatmap = cv2.resize(heatmap, (config.IMG_WIDTH, config.IMG_HEIGHT))
    colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    original_bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(original_bgr, 0.6, colored, 0.4, 0)

    Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)).save(output_path)
    return output_path