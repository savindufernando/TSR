from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class DatasetConfig:
    img_size: int = 224
    batch_size: int = 64
    seed: int = 1337
    val_split: float = 0.15
    shuffle_buffer: int = 2048


def _find_train_dir(dataset_root: Path) -> Path:
    candidates = [dataset_root / "Train", dataset_root / "train"]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find Train/ under: {dataset_root}")


def _find_test_dir(dataset_root: Path) -> Optional[Path]:
    candidates = [dataset_root / "Test", dataset_root / "test"]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def build_augmentation() -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomTranslation(0.08, 0.08),
            tf.keras.layers.RandomContrast(0.2),
            tf.keras.layers.Lambda(
                lambda x: tf.clip_by_value(
                    x + tf.random.normal(tf.shape(x), mean=0.0, stddev=0.03), 0.0, 1.0
                )
            ),
        ],
        name="augmentation",
    )


def build_normalization() -> tf.keras.Model:
    return tf.keras.Sequential(
        [tf.keras.layers.Rescaling(1.0 / 255.0)],
        name="normalize",
    )


def _dataset_from_directory(
    directory: Path,
    config: DatasetConfig,
    subset: Optional[str] = None,
) -> tf.data.Dataset:
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        image_size=(config.img_size, config.img_size),
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
        validation_split=config.val_split if subset in {"training", "validation"} else None,
        subset=subset,
    )


def _dataset_from_test_csv(
    dataset_root: Path,
    csv_path: Path,
    config: DatasetConfig,
    num_classes: int,
) -> tf.data.Dataset:
    image_paths: list[str] = []
    labels: list[int] = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = {name.lower(): name for name in (reader.fieldnames or [])}

        path_key = fieldnames.get("path") or fieldnames.get("filename")
        class_key = (
            fieldnames.get("classid")
            or fieldnames.get("class_id")
            or fieldnames.get("label")
            or fieldnames.get("class")
        )
        if not path_key or not class_key:
            raise ValueError(
                f"Unsupported Test.csv schema: fields={reader.fieldnames}; "
                "expected Path/Filename and ClassId/Label."
            )

        for row in reader:
            image_paths.append(str(dataset_root / row[path_key]))
            labels.append(int(row[class_key]))

    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.one_hot(labels, depth=num_classes))

    def _load_image(path: tf.Tensor) -> tf.Tensor:
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (config.img_size, config.img_size))
        img = tf.cast(img, tf.float32)
        return img

    image_ds = path_ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = tf.data.Dataset.zip((image_ds, label_ds))
    ds = ds.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def load_gtsrb_datasets(
    dataset_root: str | Path,
    config: Optional[DatasetConfig] = None,
    *,
    apply_preprocessing: bool = True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Optional[tf.data.Dataset], int]:
    """
    Returns: (train_ds, val_ds, test_ds_or_none, num_classes)

    Works with the KaggleHub dataset layout:
    - Train/<class_id>/*.png
    - Test/<class_id>/*.png  OR  Test.csv + images
    """
    config = config or DatasetConfig()
    dataset_root = Path(dataset_root)

    train_dir = _find_train_dir(dataset_root)
    train_ds = _dataset_from_directory(train_dir, config, subset="training")
    val_ds = _dataset_from_directory(train_dir, config, subset="validation")

    class_names = getattr(train_ds, "class_names", None) or []
    num_classes = len(class_names) if class_names else 43

    test_ds: Optional[tf.data.Dataset] = None
    test_dir = _find_test_dir(dataset_root)
    if test_dir is not None:
        has_class_subdirs = any(p.is_dir() for p in test_dir.iterdir())
        if has_class_subdirs:
            test_ds = _dataset_from_directory(test_dir, config, subset=None)

    if test_ds is None:
        csv_candidates = [dataset_root / "Test.csv", dataset_root / "test.csv"]
        for csv_path in csv_candidates:
            if csv_path.exists():
                test_ds = _dataset_from_test_csv(dataset_root, csv_path, config, num_classes)
                break

    if apply_preprocessing:
        normalize = build_normalization()
        augment = build_augmentation()

        def _prep_train(x: tf.Tensor, y: tf.Tensor):
            x = normalize(x)
            x = augment(x)
            return x, y

        def _prep_eval(x: tf.Tensor, y: tf.Tensor):
            x = normalize(x)
            return x, y

        train_ds = train_ds.map(_prep_train, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(_prep_eval, num_parallel_calls=tf.data.AUTOTUNE)
        if test_ds is not None:
            test_ds = test_ds.map(_prep_eval, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, test_ds, num_classes


def take_representative_batches(ds: tf.data.Dataset, max_batches: int = 100):
    for i, (x, _) in enumerate(ds):
        if i >= max_batches:
            return
        yield [x]
