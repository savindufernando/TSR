from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf


def _load_class_names(class_names_path: Path | None) -> List[str]:
    if class_names_path is None:
        return []
    lines = class_names_path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def _load_image(path: Path, img_size: int) -> tf.Tensor:
    img_bytes = tf.io.read_file(str(path))

    try:
        img = tf.image.decode_jpeg(img_bytes, channels=3)
    except tf.errors.InvalidArgumentError:
        img = tf.image.decode_png(img_bytes, channels=3)

    img = tf.image.resize(img, (img_size, img_size))
    img = tf.cast(img, tf.float32) / 255.0
    return img


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run inference on a single image with a SavedModel."
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to SavedModel directory or .keras/.h5 file")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--img-size", type=int, default=224,
                        help="Image size to resize to")
    parser.add_argument("--top-k", type=int, default=5,
                        help="How many top classes to display")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Minimum confidence threshold to display a class")
    parser.add_argument("--class-names", type=str,
                        help="Optional text file with class names")

    args = parser.parse_args()

    model_path = Path(args.model)
    image_path = Path(args.image)

    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    if image_path.suffix.lower() not in valid_suffixes:
        print(f"Warning: Image suffix '{image_path.suffix}' might not be supported. "
              f"Expected one of: {valid_suffixes}")

    model = tf.keras.models.load_model(model_path, compile=False)
    class_names = _load_class_names(Path(args.class_names)) if args.class_names else []

    image = _load_image(image_path, args.img_size)
    logits = model(tf.expand_dims(image, 0), training=False)[0].numpy()

    if logits.ndim != 1:
        raise RuntimeError("Model output is not a 1D class vector.")

    probs = tf.nn.softmax(logits).numpy()

    if class_names and len(class_names) != probs.shape[0]:
        print(f"Warning: {len(class_names)} class names provided but model outputs {probs.shape[0]} classes.")

    top_k = min(int(args.top_k), probs.shape[0])
    top_indices = np.argsort(probs)[-top_k:][::-1]

    print(f"Image: {args.image}")
    print(f"Model: {args.model}")

    for rank, idx in enumerate(top_indices, start=1):
        if probs[idx] < args.threshold:
            continue
        name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        print(f"{rank}: {name} (idx={idx}) prob={probs[idx]:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
