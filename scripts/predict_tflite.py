from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf


def _load_class_names(class_names_path: Path | None) -> List[str]:
    if class_names_path is None:
        return []
    return [line.strip() for line in class_names_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_image(path: Path, img_size: int) -> np.ndarray:
    img = tf.io.read_file(str(path))
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size))
    img = tf.cast(img, tf.float32) / 255.0
    return img.numpy()


def _prepare_input(img: np.ndarray, input_details: dict) -> np.ndarray:
    img = np.expand_dims(img, 0)  # add batch
    dtype = input_details["dtype"]
    scale, zero_point = input_details.get("quantization", (0.0, 0))

    if np.issubdtype(dtype, np.integer):
        scale = scale or 1.0
        img = img / scale + zero_point
        min_val, max_val = np.iinfo(dtype).min, np.iinfo(dtype).max
        img = np.clip(np.rint(img), min_val, max_val).astype(dtype)
    else:
        img = img.astype(dtype)
    return img


def main() -> int:
    parser = argparse.ArgumentParser(description="Run inference on a single image with a TFLite model.")
    parser.add_argument("--model", type=str, required=True, help="Path to .tflite file.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--img-size", type=int, default=224, help="Image size to resize to.")
    parser.add_argument("--top-k", type=int, default=5, help="How many top classes to display.")
    parser.add_argument("--class-names", type=str, help="Optional text file with class names (one per line).")
    args = parser.parse_args()

    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    class_names = _load_class_names(Path(args.class_names) if args.class_names else None)

    img = _load_image(Path(args.image), args.img_size)
    input_tensor = _prepare_input(img, input_details)

    interpreter.set_tensor(input_details["index"], input_tensor)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details["index"])[0]

    if np.issubdtype(preds.dtype, np.integer):
        scale, zero_point = output_details.get("quantization", (1.0, 0))
        preds = scale * (preds.astype(np.float32) - zero_point)

    top_k = int(args.top_k)
    top_indices = preds.argsort()[-top_k:][::-1]

    print(f"Image: {args.image}")
    print(f"Model: {args.model}")
    for rank, idx in enumerate(top_indices, start=1):
        name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        prob = float(preds[idx])
        print(f"{rank}: {name} (idx={idx}) score={prob:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
