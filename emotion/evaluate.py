"""量化模型评估：准确率、业务分组准确率、关键混淆方向。

    uv run python -m emotion.evaluate
"""

import argparse

import numpy as np

from emotion.config import CLASS_NAMES, DOCS_DIR, TFLITE_PATH
from emotion.data import load_test_dataset
from emotion.plots import plot_confusion_matrix

# 业务分组：surprise / {fear, disgust, sad} / happy / anger / neutral
BUSINESS_GROUP = np.array([0, 1, 1, 2, 1, 3, 4])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="评估 INT8 量化模型")
    p.add_argument("--model", default=str(TFLITE_PATH))
    return p.parse_args(argv)


def make_interpreter(path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter
    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    return interpreter


def main(argv=None):
    args = parse_args(argv)
    interpreter = make_interpreter(args.model)

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    output_scale, output_zero_point = output_detail.get("quantization", (0.0, 0))

    y_true, y_pred = [], []
    for images, labels in load_test_dataset():
        batch = np.clip(np.rint(images.numpy() * 255.0), 0, 255).astype(np.uint8)
        for i in range(batch.shape[0]):
            interpreter.set_tensor(input_detail["index"], batch[i : i + 1])
            interpreter.invoke()
            out = interpreter.get_tensor(output_detail["index"])[0].astype(np.float32)
            if output_scale:
                out = (out - output_zero_point) * output_scale
            y_pred.append(int(np.argmax(out)))
        y_true.extend(labels.numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = float(np.mean(y_pred == y_true) * 100)
    business = float(np.mean(BUSINESS_GROUP[y_pred] == BUSINESS_GROUP[y_true]) * 100)

    mild_true = np.isin(y_true, [1, 2, 4])
    mild_pred = np.isin(y_pred, [1, 2, 4])
    mild_recall = float(np.mean(mild_pred[mild_true]) * 100) if mild_true.any() else 0.0
    mild_to_neutral = float(np.mean(y_pred[mild_true] == 6) * 100) if mild_true.any() else 0.0
    happy_true = y_true == 3
    happy_to_anger = float(np.mean(y_pred[happy_true] == 5) * 100) if happy_true.any() else 0.0
    neutral_true = y_true == 6
    neutral_to_mild = float(np.mean(mild_pred[neutral_true]) * 100) if neutral_true.any() else 0.0

    print(f"测试集样本数:            {len(y_true)}")
    print(f"7 类准确率:              {accuracy:.2f}%")
    print(f"业务分组准确率:          {business:.2f}%")
    print(f"轻度负向组召回:          {mild_recall:.2f}%")
    print(f"轻度负向 -> neutral 泄漏: {mild_to_neutral:.2f}%")
    print(f"neutral -> 轻度负向误报:  {neutral_to_mild:.2f}%")
    print(f"happy -> anger 泄漏:      {happy_to_anger:.2f}%")

    out = plot_confusion_matrix(y_true, y_pred, CLASS_NAMES, DOCS_DIR / "confusion_matrix.png")
    print(f"混淆矩阵已保存: {out}")


if __name__ == "__main__":
    main()
