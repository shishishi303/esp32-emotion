"""模型结构：48x48x1 输入，7 类 softmax 输出。

默认配置（16/32/64/128）参数量 106,583，约 8.3 M MACs，
INT8 量化后约 118 KB，适配 ESP32-S3 的内存与算力预算。

可选变体：
  pooling   —— max / avg 池化互换（池化不计 MACs）
  dual_pool —— GAP + GMP 拼接后接分类头（多一次 max 归约，可忽略）
  se_ratio  —— 最后卷积后的轻量 SE 通道注意力压缩比
  fpn       —— 融合中高层多尺度特征
  business_head —— 训练期增加五组辅助分类头，导出时移除
"""

from tensorflow import keras
from tensorflow.keras import layers

from emotion.config import CLASS_NAMES, IMG_SHAPE

# 默认通道配置，对应原始的 106,583 参数模型
DEFAULT_CHANNELS = (16, 32, 64, 128)


def conv_block(inputs, filters: int, pooling: str = "max"):
    """Conv -> BN -> ReLU6 -> Pool。"""
    x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)
    if pooling == "avg":
        x = layers.AveragePooling2D((2, 2))(x)
    else:
        x = layers.MaxPooling2D((2, 2))(x)
    return x


def build_model(
    num_classes: int = len(CLASS_NAMES),
    name: str = "pupil",
    dropout_head: float = 0.10,
    dropout_fc: float = 0.05,
    channels=DEFAULT_CHANNELS,
    head_units: int = 64,
    pooling: str = "avg",
    dual_pool: bool = False,
    se_ratio: int | None = None,
    fpn: bool = False,
    business_head: bool = False,
) -> keras.Model:
    c1, c2, c3, c4 = channels

    inputs = keras.Input(shape=IMG_SHAPE, name="image")

    x = conv_block(inputs, c1, pooling=pooling)
    x = conv_block(x, c2, pooling=pooling)
    mid_block = x
    x = conv_block(x, c3, pooling=pooling)
    last_block = x

    x = layers.Conv2D(c4, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)

    if fpn:
        mid_projection = layers.AveragePooling2D((2, 2))(mid_block)
        mid_projection = layers.Conv2D(64, (1, 1), padding="same", use_bias=False, name="fpn_mid_conv")(mid_projection)
        mid_projection = layers.BatchNormalization(name="fpn_mid_bn")(mid_projection)
        mid_projection = layers.ReLU(max_value=6.0)(mid_projection)

        last_projection = layers.Conv2D(64, (1, 1), padding="same", use_bias=False, name="fpn_last_conv")(last_block)
        last_projection = layers.BatchNormalization(name="fpn_last_bn")(last_projection)
        last_projection = layers.ReLU(max_value=6.0)(last_projection)

        high_projection = layers.Conv2D(64, (1, 1), padding="same", use_bias=False, name="fpn_high_conv")(x)
        high_projection = layers.BatchNormalization(name="fpn_high_bn")(high_projection)
        high_projection = layers.ReLU(max_value=6.0)(high_projection)
        x = layers.Concatenate()([mid_projection, last_projection, high_projection])

    if se_ratio is not None:
        if not 1 <= se_ratio <= c4:
            raise ValueError(f"se_ratio 必须在 1 到 {c4} 之间")
        scale = layers.GlobalAveragePooling2D()(x)
        scale = layers.Dense(c4 // se_ratio, activation="relu")(scale)
        scale = layers.Dense(c4, activation="sigmoid")(scale)
        scale = layers.Reshape((1, 1, c4))(scale)
        x = layers.Multiply()([x, scale])

    pooled = layers.GlobalAveragePooling2D()(x)
    if dual_pool:
        pooled = layers.Concatenate()([pooled, layers.GlobalMaxPooling2D()(x)])

    pooled = layers.Dropout(dropout_head)(pooled)
    pooled = layers.Dense(head_units)(pooled)
    pooled = layers.ReLU(max_value=6.0)(pooled)
    pooled = layers.Dropout(dropout_fc)(pooled)

    outputs = layers.Dense(num_classes, activation="softmax", name="probs")(pooled)
    if business_head:
        group_outputs = layers.Dense(5, activation="softmax", name="business_probs")(pooled)
        return keras.Model(inputs, [outputs, group_outputs], name=name)
    return keras.Model(inputs, outputs, name=name)
