from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score

from tsr.data import DatasetConfig, load_gtsrb_datasets


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained GTSRB model.")
    parser.add_argument("--data", type=str, required=True, help="Dataset root.")
    parser.add_argument("--model", type=str, required=True, help="SavedModel directory or .keras file.")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)

    
    args = parser.parse_args()

    data_cfg = DatasetConfig(img_size=args.img_size, batch_size=args.batch_size)
    _, _, test_ds, num_classes = load_gtsrb_datasets(args.data, data_cfg)
    if test_ds is None:
        raise SystemExit("No test set found (expected Test/ or Test.csv).")

    model = tf.keras.models.load_model(args.model)

    y_true: list[int] = []
    y_pred: list[int] = []
    for x, y in test_ds:
        probs = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1).tolist())
        y_true.extend(np.argmax(y.numpy(), axis=1).tolist())

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            digits=4,
            zero_division=0,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

