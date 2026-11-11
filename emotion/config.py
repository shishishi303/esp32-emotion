"""集中配置：路径、超参数、类别定义。

所有模块都从这里取常量，避免同一条规则在多个文件里写死两遍。
"""

from pathlib import Path

# emotion/config.py -> 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"
FIRMWARE_DIR = PROJECT_ROOT / "firmware"

# 输入：48x48 灰度图
IMG_SIZE = (48, 48)
IMG_SHAPE = (*IMG_SIZE, 1)

BATCH_SIZE = 64
EPOCHS = 120
LEARNING_RATE = 1e-3
SEED = 42

# 训练集里切出 15% 做验证；test 只在最后评估时用一次
VALIDATION_SPLIT = 0.15

# 类别索引必须和 data/train 下的子目录名 0-6 严格对应。
# 顺序一旦错位，loss 里的业务惩罚项会作用到错误的类别上，而且不会报错。
CLASS_NAMES = ["surprise", "fear", "disgust", "happy", "sad", "anger", "neutral"]

SURPRISE, FEAR, DISGUST, HAPPY, SAD, ANGER, NEUTRAL = range(7)

# 类别权重，缓解样本不平衡（happy 约占训练集 39%，fear 仅 2%）
CLASS_WEIGHTS = {
    SURPRISE: 1.05,
    FEAR: 1.18,
    DISGUST: 1.22,
    HAPPY: 1.35,
    SAD: 1.20,
    ANGER: 1.20,
    NEUTRAL: 1.05,
}

# ---- 模型产物 ----
MODEL_NAME = "pupil"
KERAS_PATH = MODELS_DIR / f"{MODEL_NAME}.keras"
TFLITE_PATH = MODELS_DIR / f"{MODEL_NAME}.tflite"
HEADER_PATH = FIRMWARE_DIR / f"{MODEL_NAME}.h"

# 头文件里的数组名与固件端保持一致，替换文件时 C++ 代码无需改动
HEADER_ARRAY_NAME = "emotion_model_quant_tflite"
