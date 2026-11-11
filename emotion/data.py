"""数据管线（Keras 3）。

原始版本用的是 `keras.preprocessing.image.ImageDataGenerator`，
该 API 在 Keras 3 中已被移除，这里用 `image_dataset_from_directory` + `tf.data`
重建，增强参数与原先保持一致：

    几何：旋转 ±10°、平移 5%、缩放 10%、水平翻转、fill_mode=nearest
    光度：亮度、对比度、高斯模糊、噪点（各自带触发概率）

验证集从 `data/train` 里切出 15%，`data/test` 只在最终评估时使用一次。
原先 train/test 混用会导致测试集参与早停与量化校准，评估结果偏乐观。
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from emotion.config import (
    BATCH_SIZE,
    CLASS_NAMES,
    DATA_DIR,
    IMG_SIZE,
    SEED,
    VALIDATION_SPLIT,
)

AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def _geometric_augmentation() -> keras.Sequential:
    """对应原 ImageDataGenerator 里的几何变换。"""
    return keras.Sequential(
        [
            keras.layers.RandomRotation(10 / 360, fill_mode="nearest"),
            keras.layers.RandomTranslation(0.05, 0.05, fill_mode="nearest"),
            keras.layers.RandomZoom(0.10, 0.10, fill_mode="nearest"),
            keras.layers.RandomFlip("horizontal"),
        ],
        name="geometric_aug",
    )


def _maybe(x, p, fn):
    """以概率 p 执行 fn —— 对应原 v3_preprocessing 里的各个 if 分支。"""
    return tf.cond(tf.random.uniform(()) < p, lambda: fn(x), lambda: x)


def _gaussian_kernel(sigma, radius: int = 2):
    """可分离的二维高斯核，形状 (2r+1, 2r+1)。TF 核心库没有现成的 gaussian_filter2d。"""
    coords = tf.cast(tf.range(-radius, radius + 1), tf.float32)
    g = tf.exp(-(coords**2) / (2.0 * sigma**2))
    g = g / tf.reduce_sum(g)
    return tf.tensordot(g, g, axes=0)


def photometric_augmentation(image):
    """原 v3_preprocessing 的等价实现，输入输出取值均为 [0, 1]。"""

    def brightness(x):
        factor = tf.random.uniform((), 0.45, 1.55)
        return tf.clip_by_value(x * factor, 0.0, 1.0)

    def contrast(x):
        factor = tf.random.uniform((), 0.85, 1.25)
        mean = tf.reduce_mean(x)
        return tf.clip_by_value((x - mean) * factor + mean, 0.0, 1.0)

    def blur(x):
        sigma = tf.random.uniform((), 0.15, 0.45)
        kernel = _gaussian_kernel(sigma)
        # depthwise_conv2d 要求 kernel 形状 (kh, kw, in_channels, depth_multiplier)
        kernel = kernel[:, :, tf.newaxis, tf.newaxis]
        blurred = tf.nn.depthwise_conv2d(
            tf.expand_dims(x, 0), kernel, strides=[1, 1, 1, 1], padding="SAME"
        )
        return tf.squeeze(blurred, 0)

    def add_noise(x):
        var = tf.random.uniform((), 0.004, 0.010)
        noise = tf.random.normal(tf.shape(x), stddev=tf.sqrt(var))
        return tf.clip_by_value(x + noise, 0.0, 1.0)

    image = _maybe(image, 0.30, brightness)
    image = _maybe(image, 0.20, contrast)
    image = _maybe(image, 0.12, blur)
    image = _maybe(image, 0.20, add_noise)
    return image


def _raw_dataset(directory, batch_size, shuffle, seed=None, subset=None):
    return keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        color_mode="grayscale",
        image_size=IMG_SIZE,
        interpolation="nearest",
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        validation_split=VALIDATION_SPLIT if subset else None,
        subset=subset,
    )


def _prepared(ds, augment: bool):
    aug = _geometric_augmentation() if augment else None

    def _map(images, labels):
        images = tf.cast(images, tf.float32)
        if aug is not None:
            images = aug(images, training=True)
        images = images / 255.0
        if aug is not None:
            images = tf.map_fn(photometric_augmentation, images, fn_output_signature=tf.float32)
        return images, labels

    return ds.map(_map, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


def _assert_class_order(ds):
    """目录名必须是 0..6 —— 顺序错位会让 loss 的惩罚项作用到错误的类别上。"""
    found = [int(name) for name in ds.class_names]
    expected = list(range(len(CLASS_NAMES)))
    if found != expected:
        raise ValueError(f"data/train 的类别子目录顺序是 {found}，期望 {expected}")


def load_datasets(data_dir=DATA_DIR, batch_size=BATCH_SIZE, seed=SEED):
    """返回 (train_ds, val_ds, test_ds)。只有 train_ds 带数据增强。"""
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    for d in (train_dir, test_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"找不到数据集目录: {d}")

    train_raw = _raw_dataset(train_dir, batch_size, True, seed, subset="training")
    _assert_class_order(train_raw)
    val_raw = _raw_dataset(train_dir, batch_size, True, seed, subset="validation")
    test_raw = _raw_dataset(test_dir, batch_size, False)

    return (
        _prepared(train_raw, augment=True),
        _prepared(val_raw, augment=False),
        _prepared(test_raw, augment=False),
    )


def load_test_dataset(data_dir=DATA_DIR, batch_size=BATCH_SIZE):
    test_dir = Path(data_dir) / "test"
    return _prepared(_raw_dataset(test_dir, batch_size, False), augment=False)


def load_calibration_paths(data_dir=DATA_DIR, per_class: int = 150, seed: int = SEED):
    """量化校准样本：每类等量抽样，来源是训练集（不是测试集）。"""
    train_dir = Path(data_dir) / "train"
    rng = np.random.default_rng(seed)
    paths = []

    for idx in range(len(CLASS_NAMES)):
        class_dir = train_dir / str(idx)
        files = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise FileNotFoundError(f"类别 {idx} 没有图片: {class_dir}")
        take = min(per_class, len(files))
        chosen = rng.choice(np.array(files, dtype=object), size=take, replace=False)
        paths.extend(Path(p) for p in chosen)

    rng.shuffle(paths)
    return paths
