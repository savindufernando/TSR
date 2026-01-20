from __future__ import annotations

import tensorflow as tf

from tsr.model import ModelConfig, build_hybrid_cnn_vit


def test_model_output_shape():
    cfg = ModelConfig(
        img_size=64,
        num_classes=5,
        token_dim=64,
        num_heads=4,
        transformer_layers=1,
        mlp_dim=128,
        dropout=0.0,
        backbone_weights=None,  # avoid downloading ImageNet weights in tests
    )
    model = build_hybrid_cnn_vit(cfg)

    dummy = tf.zeros((2, cfg.img_size, cfg.img_size, 3), dtype=tf.float32)
    preds = model(dummy, training=False)

    assert preds.shape == (2, cfg.num_classes)
