"""模型结构：48x48x1 输入，7 类 softmax 输出。

默认配置（16/32/64/128）参数量约 10.6 万，INT8 量化后约 120 KB，
适配 ESP32-S3 的内存预算；channels 参数可以放大容量以换取精度。
"""

from tensorflow import keras
from tensorflow.keras import layers

from emotion.config import CLASS_NAMES, IMG_SHAPE

# 默认通道配置，对应原始的 106,583 参数模型
DEFAULT_CHANNELS = (16, 32, 64, 128)


def conv_block(inputs, filters: int):
    """Conv -> BN -> ReLU6 -> MaxPool。"""
    x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    return x


def build_model(
    num_classes: int = len(CLASS_NAMES),
    name: str = "pupil",
    dropout_head: float = 0.24,
    dropout_fc: float = 0.17,
    channels=DEFAULT_CHANNELS,
    head_units: int = 64,
) -> keras.Model:
    c1, c2, c3, c4 = channels

    inputs = keras.Input(shape=IMG_SHAPE, name="image")

    x = conv_block(inputs, c1)
    x = conv_block(x, c2)
    x = conv_block(x, c3)

    x = layers.Conv2D(c4, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(dropout_head)(x)
    x = layers.Dense(head_units)(x)
    x = layers.ReLU(max_value=6.0)(x)
    x = layers.Dropout(dropout_fc)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="probs")(x)
    return keras.Model(inputs, outputs, name=name)
