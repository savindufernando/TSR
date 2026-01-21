from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from tsr.data import DatasetConfig, load_gtsrb_datasets, take_representative_batches


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a SavedModel to TFLite.")
    parser.add_argument("--saved-model", dest="saved_model", type=str, required=True,
                        help="SavedModel directory")
    parser.add_argument("--out", type=str, required=True,
                        help="Output .tflite path")
    parser.add_argument("--int8", action="store_true",
                        help="Enable full integer quantization")
    parser.add_argument("--data", type=str, default=None,
                        help="Dataset root (required for --int8)")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)

    args = parser.parse_args()

    saved_model_path = Path(args.saved_model)
    if not saved_model_path.exists():
        raise SystemExit(f"SavedModel not found: {saved_model_path}")

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Allow fallback to SELECT_TF_OPS for layers not natively in TFLite (like some ViT ops)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    if args.int8:
        if not args.data:
            raise SystemExit("--data is required when using --int8.")

        data_root = Path(args.data)
        if not data_root.exists():
            raise SystemExit(f"Dataset root not found: {data_root}")

        train_ds, _, _, _ = load_gtsrb_datasets(
            data_root,
            DatasetConfig(
                img_size=args.img_size,
                batch_size=args.batch_size,
            ),
            apply_preprocessing=False,
        )

        def representative_dataset():
            for batch in take_representative_batches(train_ds, max_batches=100):
                yield [tf.cast(batch[0], tf.float32)]

        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_model)

    print(f"Successfully exported TFLite model to: {out_path.absolute()}")
    print(f"Model size: {out_path.stat().st_size / 1024 / 1024:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
