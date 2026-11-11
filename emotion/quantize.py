"""INT8 全量化：FP32 Keras 模型 -> TFLite。

    uv run python -m emotion.quantize
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from emotion.config import IMG_SIZE, KERAS_PATH, TFLITE_PATH
from emotion.data import load_calibration_paths


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="INT8 量化")
    p.add_argument("--model", default=str(KERAS_PATH), help="输入 .keras 模型")
    p.add_argument("--output", default=str(TFLITE_PATH), help="输出 .tflite")
    p.add_argument("--per-class", type=int, default=150, help="每类校准样本数")
    return p.parse_args(argv)


def representative_dataset(paths):
    for path in paths:
        img = keras.utils.load_img(path, color_mode="grayscale", target_size=IMG_SIZE)
        arr = keras.utils.img_to_array(img).astype(np.float32) / 255.0
        yield [np.expand_dims(arr, axis=0)]


def main(argv=None):
    args = parse_args(argv)

    model = keras.models.load_model(args.model, compile=False)

    paths = load_calibration_paths(per_class=args.per_class)
    print(f"校准样本: {len(paths)} 张（来自训练集，每类 <= {args.per_class}）")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(paths)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8

    tflite_model = converter.convert()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(tflite_model)
    print(f"量化模型已保存: {out}  ({len(tflite_model) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
