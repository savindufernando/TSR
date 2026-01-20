from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from tsr.data import DatasetConfig, compute_class_weights, load_gtsrb_datasets
from tsr.model import ModelConfig, build_hybrid_cnn_vit


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Hybrid CNN+ViT model for GTSRB.")
    parser.add_argument("--data", type=str, required=True, help="Dataset root (contains Train/).")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing factor.")
    parser.add_argument("--backbone-trainable", action="store_true")
    parser.add_argument("--out", type=str, default="outputs", help="Output folder.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility.")
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Cache train/val/test datasets in memory (useful for smaller datasets).",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable data augmentation (keeps only normalization).",
    )
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Compute class weights from Train/ and pass to model.fit (mitigates imbalance).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    data_cfg = DatasetConfig(
        img_size=args.img_size,
        batch_size=args.batch_size,
        seed=args.seed,
        cache=bool(args.cache),
        augment=not bool(args.no_augment),
    )
    train_ds, val_ds, test_ds, num_classes = load_gtsrb_datasets(args.data, data_cfg)

    model_cfg = ModelConfig(
        img_size=args.img_size,
        num_classes=num_classes,
        backbone_trainable=bool(args.backbone_trainable),
    )
    model = build_hybrid_cnn_vit(model_cfg)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=float(args.label_smoothing)),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "checkpoints" / "best.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=str(out_dir / "logs")),
    ]
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(args.data, num_classes=num_classes)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    saved_model_dir = out_dir / "saved_model"
    model.save(saved_model_dir)
    print("SavedModel:", saved_model_dir)

    if test_ds is not None:
        results = model.evaluate(test_ds, verbose=2)
        print("Test metrics:", dict(zip(model.metrics_names, results)))
    else:
        print("No test set detected; skipping test evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
