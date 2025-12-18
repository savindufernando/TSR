from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import tensorflow as tf


@dataclass(frozen=True)
class ModelConfig:
    img_size: int = 224
    num_classes: int = 43
    backbone_trainable: bool = False
    token_dim: int = 256
    num_heads: int = 4
    transformer_layers: int = 4
    mlp_dim: int = 512
    dropout: float = 0.1


def _transformer_encoder(
    x: tf.Tensor,
    *,
    num_heads: int,
    token_dim: int,
    mlp_dim: int,
    dropout: float,
    name: str,
) -> tf.Tensor:
    attn_in = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln1")(x)
    attn_out = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=token_dim // num_heads,
        dropout=dropout,
        name=f"{name}_mha",
    )(attn_in, attn_in)
    x = tf.keras.layers.Add(name=f"{name}_attn_add")([x, attn_out])

    mlp_in = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln2")(x)
    mlp_out = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(mlp_dim, activation=tf.nn.gelu),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(token_dim),
            tf.keras.layers.Dropout(dropout),
        ],
        name=f"{name}_mlp",
    )(mlp_in)
    x = tf.keras.layers.Add(name=f"{name}_mlp_add")([x, mlp_out])
    return x


def build_hybrid_cnn_vit(config: Optional[ModelConfig] = None) -> tf.keras.Model:
    config = config or ModelConfig()
    if config.token_dim % config.num_heads != 0:
        raise ValueError("token_dim must be divisible by num_heads")

    inputs = tf.keras.Input(shape=(config.img_size, config.img_size, 3), name="image")

    backbone = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
    )
    backbone.trainable = bool(config.backbone_trainable)

    feature_map = backbone.output  # (B, H, W, C)
    fm_h = feature_map.shape[1]
    fm_w = feature_map.shape[2]
    c = feature_map.shape[-1]
    if fm_h is None or fm_w is None or c is None:
        raise ValueError("Backbone feature map shape must be static; use a fixed img_size.")

    tokens = tf.keras.layers.Reshape((-1, c), name="flatten_tokens")(feature_map)
    tokens = tf.keras.layers.Dense(config.token_dim, name="token_projection")(tokens)

    num_tokens = int(fm_h) * int(fm_w)
    pos_embed = tf.keras.layers.Embedding(
        input_dim=num_tokens,
        output_dim=config.token_dim,
        name="pos_embedding",
    )
    positions = tf.range(start=0, limit=num_tokens, delta=1)
    tokens = tokens + pos_embed(positions)

    x = tf.keras.layers.Dropout(config.dropout, name="token_dropout")(tokens)
    for i in range(config.transformer_layers):
        x = _transformer_encoder(
            x,
            num_heads=config.num_heads,
            token_dim=config.token_dim,
            mlp_dim=config.mlp_dim,
            dropout=config.dropout,
            name=f"enc{i+1}",
        )

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="pre_head_ln")(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name="token_gap")(x)
    x = tf.keras.layers.Dropout(config.dropout, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(config.num_classes, activation="softmax", name="pred")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="hybrid_mobilenetv2_transformer")
    return model
