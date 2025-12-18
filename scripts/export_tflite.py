from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from tsr.data import DatasetConfig, load_gtsrb_datasets, take_representative_batches


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a SavedModel to TFLite.")
    parser.add_argument("--saved-model", type=str, required=True, help="SavedModel directory.")
    parser.add_argument("--out", type=str, required=True, help="Output .tflite path.")
    parser.add_argument("--int8", action="store_true", help="Enable full integer quantization.")
    parser.add_argument("--data", type=str, default=None, help="Dataset root (required for --int8).")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    converter = tf.lite.TFLiteConverter.from_saved_model(args.saved_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if args.int8:
        if not args.data:
            raise SystemExit("--data is required when using --int8 (representative dataset).")
        train_ds, _, _, _ = load_gtsrb_datasets(
            args.data,
            DatasetConfig(img_size=args.img_size, batch_size=args.batch_size),
            apply_preprocessing=False,
        )
        converter.representative_dataset = lambda: take_representative_batches(train_ds, max_batches=100)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_model)
    print("Wrote:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
