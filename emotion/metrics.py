"""分类与业务指标。"""

import numpy as np

from emotion.config import BUSINESS_GROUP as BUSINESS_GROUP_INDICES, CLASS_NAMES

BUSINESS_GROUP = np.asarray(BUSINESS_GROUP_INDICES)
BUSINESS_CLASS_NAMES = ["surprise", "mild_negative", "happy", "anger", "neutral"]


def metrics_from_confusion_matrix(cm):
    cm = np.asarray(cm, dtype=np.int64)
    business_cm = np.zeros((5, 5), dtype=np.int64)
    for true in range(len(CLASS_NAMES)):
        for pred in range(len(CLASS_NAMES)):
            business_cm[BUSINESS_GROUP[true], BUSINESS_GROUP[pred]] += cm[true, pred]

    recalls = {}
    for idx, name in enumerate(CLASS_NAMES):
        total = int(cm[idx].sum())
        recalls[name] = float(cm[idx, idx] / total) if total else None

    group_recalls = {}
    group_precisions = {}
    for idx, name in enumerate(BUSINESS_CLASS_NAMES):
        true_total = int(business_cm[idx].sum())
        pred_total = int(business_cm[:, idx].sum())
        group_recalls[name] = float(business_cm[idx, idx] / true_total) if true_total else None
        group_precisions[name] = float(business_cm[idx, idx] / pred_total) if pred_total else None

    mild = BUSINESS_CLASS_NAMES.index("mild_negative")
    happy = BUSINESS_CLASS_NAMES.index("happy")
    anger = BUSINESS_CLASS_NAMES.index("anger")
    neutral = BUSINESS_CLASS_NAMES.index("neutral")
    surprise = BUSINESS_CLASS_NAMES.index("surprise")
    mild_total = int(business_cm[mild].sum())
    mild_recall = group_recalls["mild_negative"]
    mild_precision = group_precisions["mild_negative"]

    def mild_to(group):
        return float(business_cm[mild, group] / mild_total) if mild_total else None

    metrics = {
        "samples": int(cm.sum()),
        "accuracy": float(np.trace(cm) / cm.sum()),
        "macro_recall": float(np.mean([value for value in recalls.values() if value is not None])),
        "per_class_recall": recalls,
        "business_accuracy": float(np.trace(business_cm) / business_cm.sum()),
        "business_macro_recall": float(
            np.mean([value for value in group_recalls.values() if value is not None])
        ),
        "business_per_group_recall": group_recalls,
        "business_per_group_precision": group_precisions,
        "business_confusion_matrix": business_cm.tolist(),
        "mild_recall": mild_recall,
        "mild_precision": mild_precision,
        "mild_f1": (
            float(2 * mild_precision * mild_recall / (mild_precision + mild_recall))
            if mild_precision is not None and mild_recall is not None and mild_precision + mild_recall > 0
            else None
        ),
        "mild_to_surprise": mild_to(surprise),
        "mild_to_happy": mild_to(happy),
        "mild_to_anger": mild_to(anger),
        "mild_to_neutral": mild_to(neutral),
        "happy_to_anger": (
            float(business_cm[happy, anger] / business_cm[happy].sum())
            if business_cm[happy].sum()
            else None
        ),
        "neutral_to_mild": (
            float(business_cm[neutral, mild] / business_cm[neutral].sum())
            if business_cm[neutral].sum()
            else None
        ),
    }
    return metrics


def pareto_assessment(metrics, reference, tolerance=0.0):
    required = (
        "accuracy",
        "business_accuracy",
        "mild_f1",
        "macro_recall",
        "mild_recall",
        "mild_precision",
        "mild_to_neutral",
        "neutral_to_mild",
        "happy_to_anger",
    )
    if any(metrics.get(key) is None or not np.isfinite(metrics[key]) for key in required):
        return {"eligible": False, "score": 0.0, "violation": float("inf")}

    strict = ("accuracy", "business_accuracy", "mild_f1")
    tolerant_higher = ("macro_recall", "mild_recall", "mild_precision")
    tolerant_lower = ("mild_to_neutral", "neutral_to_mild", "happy_to_anger")
    eligible = all(metrics[key] > reference[key] for key in strict)
    eligible = eligible and all(
        metrics[key] >= reference[key] - tolerance for key in tolerant_higher
    )
    eligible = eligible and all(
        metrics[key] <= reference[key] + tolerance for key in tolerant_lower
    )
    violations = [max(reference[key] - metrics[key], 0.0) for key in strict]
    violations.extend(
        max(reference[key] - tolerance - metrics[key], 0.0) for key in tolerant_higher
    )
    violations.extend(
        max(metrics[key] - reference[key] - tolerance, 0.0) for key in tolerant_lower
    )
    return {
        "eligible": eligible,
        "score": float((metrics["business_accuracy"] + metrics["mild_f1"]) / 2),
        "violation": float(sum(violations)),
    }


def classification_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    count = len(CLASS_NAMES)
    cm = np.bincount(y_true * count + y_pred, minlength=count * count).reshape(count, count)
    return metrics_from_confusion_matrix(cm), cm
