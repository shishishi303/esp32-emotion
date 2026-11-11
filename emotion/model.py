"""模型结构：48x48x1 输入，7 类 softmax 输出。

参数量约 10.6 万，INT8 量化后约 120 KB，适配 ESP32-S3 的 RAM 预算。
"""

from tensorflow import keras
from tensorflow.keras import layers

from emotion.config import CLASS_NAMES, IMG_SHAPE


def conv_block(inputs, filters: int):
    """Conv -> BN -> ReLU6 -> MaxPool。"""
    x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    return x


def build_model(num_classes: int = len(CLASS_NAMES), name: str = "pupil") -> keras.Model:
    inputs = keras.Input(shape=IMG_SHAPE, name="image")

    x = conv_block(inputs, 16)
    x = conv_block(x, 32)
    x = conv_block(x, 64)

    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.24)(x)
    x = layers.Dense(64)(x)
    x = layers.ReLU(max_value=6.0)(x)
    x = layers.Dropout(0.17)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="probs")(x)
    return keras.Model(inputs, outputs, name=name)
