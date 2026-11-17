# esp32-emotion — ESP32-S3 上的 7 类表情识别

两个 INT8 模型，输入都是 48×48 灰度人脸，输出 7 类表情概率：

| | 体积 | 算力 | 测试集准确率 |
|---|---|---|---|
| `pupil` | 118 KB | 8.3 M MACs | 77.77% |
| `pupil-HighCalculate` | 158 KB | 15.3 M MACs | **79.43%** |

同一条训练、量化、C 头文件导出流水线。选哪个取决于板子上的时间预算。

An INT8 7-class facial-expression classifier (48×48 grayscale) for ESP32-S3,
in two sizes: 118 KB / 8.3 M MACs and 158 KB / 15.3 M MACs.

## 效果

测试集是 RAF-DB 官方划分的 3,068 张，只在最后评估一次。

| 模型 | INT8 体积 | 参数量 | MACs | 7 类准确率 | 业务分组 |
|---|---|---|---|---|---|
| `pupil` | 117.7 KB | 106,583 | 8.30 M | 77.77% | 79.11% |
| **`pupil-HighCalculate`** | **158.3 KB** | **142,037** | **15.29 M** | **79.43%** | **81.16%** |

业务指标（INT8）：

| 指标 | `pupil` | `pupil-HighCalculate` |
|---|---|---|
| 轻度负向组召回 | 58.85% | **65.17%** |
| 轻度负向精确率 | 80.89% | **81.98%** |
| 轻度负向 F1 | 68.13% | **72.61%** |
| 轻度负向 → neutral 泄漏 | 18.26% | **17.84%** |
| neutral → 轻度负向误报 | 7.65% | **7.35%** |
| happy → anger 泄漏 | 0.76% | **0.68%** |

`pupil-HighCalculate` 逐类召回：surprise 76.0%、fear 39.2%、disgust 36.3%、happy 92.6%、
sad 67.8%、anger 74.1%、neutral 82.2%。混淆矩阵见 `docs/confusion_matrix.png`。

量化损失：FP32 79.50% → INT8 79.43%，0.07 个点。

## 怎么选

| 场景 | 选择 |
|---|---|
| 采样间隔 ≥ 200 ms、和其他任务共存 | `pupil-HighCalculate`，情绪推理占 13–19% CPU |
| 采样间隔 100 ms 左右 | 两个都能跑，HighCalculate 约占 26–38% |
| 每帧都要推理（33 ms 预算） | `pupil`，HighCalculate 会掉帧 |
| 只关心轻度负向是否漏判 | `pupil-HighCalculate`，召回高 6.3 个点 |

## 方法

**输入 48×48 单通道灰度。** 上游人脸检测流程原生输出这个尺寸，改分辨率要连固件一起改。试过 96×96，在 ESP32-S3 上更臃肿，精度也没跟着涨。

**主干**：4 个 conv 块 + 平均池化 + [GAP, GMP] 拼接 + 两层全连接。两个模型的差别只在通道数：

| | conv1 | conv2 | conv3 | conv4 |
|---|---|---|---|---|
| `pupil` | 16 | 32 | 64 | 128 |
| `pupil-HighCalculate` | **32** | **48** | 64 | 128 |

加宽的是前两层。代价也集中在 conv2——它从 2.65 M 涨到 7.96 M MACs，占新模型总算力的 52%。

**`pupil-HighCalculate` 额外有一层成对校准头**：利用 happy / anger 的分数差，对这两个易混类做一次非线性的输出校正，不改变主干特征。实现是几个固定常数的全连接层（Dense + SUB + CONCAT），参数约 3.5 万，计算量可忽略。

**损失**：label smoothing 打底，叠加三条业务约束：

1. fear / disgust / sad 同属“轻度负向”，允许彼此混淆（组内软标签 0.80 / 0.09）；
2. happy 被判成 anger，代价额外放大 0.35；
3. neutral 被判成任何有表情的类别，代价放大 0.08。

实现在 `emotion/losses.py`。

## 部署到 ESP32-S3

两个头文件都在 `firmware/`，`xxd -i` 格式，数组名固定：

```c
const unsigned int emotion_model_quant_tflite_len = 162112;
const unsigned char emotion_model_quant_tflite[] = { ... };
```

接口两边一致：输入 `uint8 [1,48,48,1]`，输出 `float32 [1,7]`。覆盖到固件的
`components/emotion_recognition/pupil.h` 即可，C++ 侧不用改。

### 算子要求

`pupil` 只需要 `CONV_2D`、`AVERAGE_POOL_2D`、`MEAN`、`FULLY_CONNECTED`、`SOFTMAX`。

`pupil-HighCalculate` 多三个，**必须显式注册**，漏一个就是运行时
`Didn't find op for builtin opcode`：

```cpp
resolver.AddConcatenation();
resolver.AddReduceMax();
resolver.AddSub();
```

### 资源

| 项目 | `pupil` | `pupil-HighCalculate` |
|---|---|---|
| 权重（flash） | 118 KB | 158 KB |
| 首层激活 | 48×48×16 = 36.9 KB | 48×48×32 = 73.7 KB |
| 估算 arena | 60–90 KB | 120–170 KB |
| 估算单次推理 | 14–21 ms | 26–38 ms |

**延迟和 arena 是估算值，尚未在 ESP32-S3 实机测量。** 上板前请用
`interpreter.arena_used_bytes()` 和 `esp_timer` 确认。只有 PC 端的 INT8 推理验证过。

## 复现

需要 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync

uv run python -m emotion.train               # 训练 pupil 基线，CPU 约 20 分钟
uv run python -m emotion.quantize            # INT8 全量化
uv run python -m emotion.evaluate            # 测试集评估 + 混淆矩阵
uv run python scripts/export_c_header.py     # 生成 firmware/*.h
```

这一套产出的是 `pupil` 基线。`pupil-HighCalculate` 由基线加两阶段微调得到（通道加宽 +
happy/anger 成对校准），属于研究代码，训练脚本未随仓库发布；仓库内提供的是构建好的
INT8 产物，指标可由 `emotion.evaluate` 复算。

训练存在随机性，重跑结果会有 ±1 个点波动；上表数字来自一次具体运行。
`docs/training_curves.png` 是基线阶段的曲线。

## 数据

[RAF-DB](https://www.whdeng.cn/RAF/model1.html) 基础情绪子集，15,339 张，官方划分 12,271 训练 / 3,068 测试。

数据集不进仓库，自行申请后按下面的结构放：

```
data/
├── train/
│   ├── 0/   0 = surprise
│   ├── 1/   1 = fear
│   ├── 2/   2 = disgust
│   ├── 3/   3 = happy
│   ├── 4/   4 = sad
│   ├── 5/   5 = anger
│   └── 6/   6 = neutral
└── test/
    └── （同上）
```

目录顺序必须和 `emotion/config.py` 的 `CLASS_NAMES` 对齐。**顺序错位不会报错**，但损失函数里的业务惩罚会打到错误的类别上，训练结果照样能出，只是全错。

## 评估协议

- 验证集从 `data/train` 切 15%（1,840 张），用于早停和学习率调度。
- 测试集 3,068 张不参与任何决策，只在 `emotion.evaluate` 里跑一次。
- 量化校准样本从训练集每类抽 150 张。

上面那些数字没有选择偏差。早期版本拿测试集当验证集用，报告值带着模型选择的乐观偏差，和现在这套不能直接对比。

## 已知局限

- 48×48 灰度丢细节，少数类召回低：`pupil-HighCalculate` 的 fear 39.2%、disgust 36.3%，
  而 happy 有 92.6%（训练集 fear 281 张、disgust 717 张、happy 4,772 张）。
- 训练数据以摆拍为主，强侧光、遮挡、戴眼镜会退化。
- 7 类输出不等于连续情绪维度，端侧建议按业务做分组决策。
- `pupil-HighCalculate` 的实机延迟与内存占用未验证。

## License

MIT，见 [LICENSE](LICENSE)。
