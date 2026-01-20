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
    img = tf.io.read_file(str(path))
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size))
    img = tf.cast(img, tf.float32) / 255.0
    return img


def main() -> int:
    parser = argparse.ArgumentParser(description="Run inference on a single image with a SavedModel.")
    parser.add_argument("--model", type=str, required=True, help="Path to SavedModel directory or .keras/.h5 file.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--img-size", type=int, default=224, help="Image size to resize to.")
    parser.add_argument("--top-k", type=int, default=5, help="How many top classes to display.")
    parser.add_argument("--class-names", type=str, help="Optional text file with class names (one per line).")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    class_names = _load_class_names(Path(args.class_names) if args.class_names else None)

    image = _load_image(Path(args.image), args.img_size)
    preds = model(tf.expand_dims(image, 0), training=False).numpy()[0]

    top_k = int(args.top_k)
    top_indices = preds.argsort()[-top_k:][::-1]

    print(f"Image: {args.image}")
    print(f"Model: {args.model}")
    for rank, idx in enumerate(top_indices, start=1):
        name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        prob = float(preds[idx])
        print(f"{rank}: {name} (idx={idx}) prob={prob:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
