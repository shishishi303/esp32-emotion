"""量化模型评估：准确率、业务分组准确率、关键混淆方向。

    uv run python -m emotion.evaluate
"""

import argparse
import json
from pathlib import Path

import numpy as np

from emotion.config import BATCH_SIZE, CLASS_NAMES, DOCS_DIR, SEED, TFLITE_PATH
from emotion.data import load_test_dataset, load_validation_dataset
from emotion.metrics import BUSINESS_CLASS_NAMES, classification_metrics
from emotion.plots import plot_confusion_matrix, plot_confusion_matrix_counts


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="评估 INT8 量化模型")
    p.add_argument("--model", default=str(TFLITE_PATH))
    p.add_argument("--output-dir", default=str(DOCS_DIR))
    p.add_argument("--split", choices=("test", "validation"), default="test")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--keras-model", help="可选：同时评估对应 FP32 Keras 模型并输出量化对比")
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

    if args.split == "validation":
        dataset = load_validation_dataset(batch_size=args.batch_size, seed=args.seed)
    else:
        dataset = load_test_dataset(batch_size=args.batch_size)

    fp32_model = None
    fp32_pred = []
    if args.keras_model:
        from tensorflow import keras

        fp32_model = keras.models.load_model(args.keras_model, compile=False)

    y_true, y_pred = [], []
    for images, labels in dataset:
        if fp32_model is not None:
            fp32_pred.extend(np.argmax(np.asarray(fp32_model.predict_on_batch(images)), axis=1).tolist())
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
    metrics, _ = classification_metrics(y_true, y_pred)

    split_name = "验证集" if args.split == "validation" else "测试集"
    print(f"{split_name}样本数:            {metrics['samples']}")
    print(f"7 类准确率:              {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro recall:            {metrics['macro_recall'] * 100:.2f}%")
    print(f"业务分组准确率:          {metrics['business_accuracy'] * 100:.2f}%")
    print(f"轻度负向组召回:          {metrics['mild_recall'] * 100:.2f}%")
    print(f"轻度负向组精确率:        {metrics['mild_precision'] * 100:.2f}%")
    print(f"轻度负向组 F1:           {metrics['mild_f1'] * 100:.2f}%")
    print(f"轻度负向 -> neutral 泄漏: {metrics['mild_to_neutral'] * 100:.2f}%")
    print(f"neutral -> 轻度负向误报:  {metrics['neutral_to_mild'] * 100:.2f}%")
    print(f"happy -> anger 泄漏:      {metrics['happy_to_anger'] * 100:.2f}%")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = plot_confusion_matrix(y_true, y_pred, CLASS_NAMES, output_dir / "confusion_matrix.png")
    business_out = plot_confusion_matrix_counts(
        metrics["business_confusion_matrix"],
        BUSINESS_CLASS_NAMES,
        output_dir / "business_confusion_matrix.png",
        title="Business groups (INT8 quantized)",
    )
    np.savetxt(
        output_dir / "business_confusion_matrix.csv",
        np.asarray(metrics["business_confusion_matrix"]),
        delimiter=",",
        fmt="%d",
    )
    metrics["model"] = str(args.model)
    metrics["split"] = args.split
    metrics_path = output_dir / f"{args.split}_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if fp32_model is not None:
        fp32_metrics, _ = classification_metrics(y_true, np.asarray(fp32_pred))
        comparison_keys = (
            "accuracy",
            "macro_recall",
            "business_accuracy",
            "mild_recall",
            "mild_precision",
            "mild_f1",
            "mild_to_neutral",
            "neutral_to_mild",
            "happy_to_anger",
        )
        comparison = {
            "split": args.split,
            "keras_model": str(args.keras_model),
            "tflite_model": str(args.model),
            "parameters": int(fp32_model.count_params()),
            "fp32_file_bytes": Path(args.keras_model).stat().st_size,
            "int8_file_bytes": Path(args.model).stat().st_size,
            "int8_size_kib": Path(args.model).stat().st_size / 1024,
            "fp32": fp32_metrics,
            "int8": metrics,
            "int8_minus_fp32": {
                key: metrics[key] - fp32_metrics[key] for key in comparison_keys
            },
        }
        comparison_path = output_dir / "quantization_comparison.json"
        comparison_path.write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"FP32 准确率: {fp32_metrics['accuracy'] * 100:.2f}%")
        print(f"量化对比已保存: {comparison_path}")
    print(f"混淆矩阵已保存: {out}")
    print(f"业务分组混淆矩阵已保存: {business_out}")
    print(f"指标已保存: {metrics_path}")


if __name__ == "__main__":
    main()
