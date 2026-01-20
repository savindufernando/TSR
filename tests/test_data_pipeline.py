from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from tsr.data import DatasetConfig, load_gtsrb_datasets


def _write_png(path: Path, size: int = 32):
    img = tf.random.uniform((size, size, 3), maxval=255, dtype=tf.int32)
    png = tf.image.encode_png(tf.cast(img, tf.uint8))
    tf.io.write_file(str(path), png)


def test_load_gtsrb_datasets(tmp_path):
    dataset_root = tmp_path / "dataset"
    for cls in ("0", "1"):
        cls_dir = dataset_root / "Train" / cls
        cls_dir.mkdir(parents=True)
        for i in range(3):
            _write_png(cls_dir / f"img_{i}.png", size=32)

    test_dir = dataset_root / "Test" / "0"
    test_dir.mkdir(parents=True)
    _write_png(test_dir / "img_test.png", size=32)

    cfg = DatasetConfig(img_size=32, batch_size=2, seed=123, cache=False)
    train_ds, val_ds, test_ds, num_classes = load_gtsrb_datasets(
        dataset_root, cfg, apply_preprocessing=False
    )

    assert num_classes == 2
    assert test_ds is not None

    batch_x, batch_y = next(iter(train_ds))
    assert batch_x.shape[1:] == (32, 32, 3)
    assert batch_y.shape[1] == num_classes

    val_batch_x, val_batch_y = next(iter(val_ds))
    assert val_batch_x.shape[1:] == (32, 32, 3)
    assert val_batch_y.shape[1] == num_classes
