"""业务感知损失函数。

普通交叉熵把 7 类当成 7 个平等的标签，但端侧表情识别有明确的业务约束：

1. fear / disgust / sad 在大类上同属"轻度负向"，彼此混淆的代价远小于
   被识别成 happy 或 neutral——用户会觉得设备"读不懂情绪"。
2. happy 被误判成 anger 是端侧最尴尬的错误，必须重罚。
3. neutral 被当成有表情，会造成"设备自作多情"的体验。

因此在 label smoothing 的基础上，给指定方向的误判加了不对称惩罚项。
"""

import tensorflow as tf
from tensorflow import keras

from emotion.config import (
    ANGER,
    BUSINESS_GROUP,
    DISGUST,
    FEAR,
    HAPPY,
    NEUTRAL,
    SAD,
    SURPRISE,
)

# 轻度负向三类：允许互相混淆
MILD_NEGATIVE = (FEAR, DISGUST, SAD)

# label smoothing 强度
SMOOTHING = 0.02
# 组内样本把多少置信度分给"兄弟类别"
GROUP_KEEP = 0.80
GROUP_SHARE = 0.09


def business_aware_loss(
    y_true,
    y_pred,
    group_keep=GROUP_KEEP,
    group_share=GROUP_SHARE,
    outside_mild_penalty=0.0,
    happy_anger_penalty=0.35,
    happy_anger_power=1.0,
    mild_neutral_penalty=0.18,
    neutral_mild_penalty=0.0,
    happy_anger_margin_weight=0.0,
    neutral_mild_margin_weight=0.0,
    pairwise_margin=0.05,
):
    """y_true: (batch,) 整数标签；y_pred: (batch, 7) softmax 概率。"""
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    num_classes = tf.shape(y_pred)[-1]
    num_classes_f = tf.cast(num_classes, y_pred.dtype)

    onehot = tf.one_hot(y_true, num_classes, dtype=y_pred.dtype)

    # ---- 基础目标：label smoothing ----
    smooth = tf.cast(SMOOTHING, y_pred.dtype)
    strict_target = onehot * (1.0 - smooth) + smooth / num_classes_f

    # ---- 轻度负向三类之间：互相"容忍" ----
    soft_mask = tf.constant([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0], dtype=y_pred.dtype)
    soft_lookup = tf.constant([False, True, True, False, True, False, False])
    in_soft = tf.gather(soft_lookup, y_true)
    peers = tf.maximum(soft_mask - onehot, 0.0)
    group_target = (
        onehot * tf.cast(group_keep, y_pred.dtype)
        + peers * tf.cast(group_share, y_pred.dtype)
        + smooth / num_classes_f
    )

    target = tf.where(tf.expand_dims(in_soft, axis=-1), group_target, strict_target)

    base_loss = keras.losses.categorical_crossentropy(target, y_pred)

    # ---- 不对称惩罚项 ----
    is_happy = tf.equal(y_true, HAPPY)
    is_anger = tf.equal(y_true, ANGER)
    is_neutral = tf.equal(y_true, NEUTRAL)

    def masked(mask, value):
        return tf.where(mask, value, tf.zeros_like(base_loss))

    penalty = (
        # happy 被判成 anger：最严重
        tf.cast(happy_anger_penalty, y_pred.dtype)
        * masked(is_happy, tf.pow(y_pred[:, ANGER], tf.cast(happy_anger_power, y_pred.dtype)))
        # happy 被判成 sad / neutral
        + 0.18 * masked(is_happy, y_pred[:, SAD] + y_pred[:, NEUTRAL])
        # anger 被判成 happy
        + 0.12 * masked(is_anger, y_pred[:, HAPPY])
        # 轻度负向被判成 neutral / happy / anger / surprise
        + tf.cast(mild_neutral_penalty, y_pred.dtype) * masked(in_soft, y_pred[:, NEUTRAL])
        + 0.10 * masked(in_soft, y_pred[:, HAPPY])
        + 0.08 * masked(in_soft, y_pred[:, ANGER])
        + 0.04 * masked(in_soft, y_pred[:, SURPRISE])
        # anger 被判成轻度负向
        + 0.06 * masked(is_anger, y_pred[:, FEAR] + y_pred[:, DISGUST] + y_pred[:, SAD])
        # neutral 被判成有表情
        + 0.08
        * masked(
            is_neutral,
            y_pred[:, SURPRISE] + y_pred[:, HAPPY] + y_pred[:, SAD] + y_pred[:, ANGER],
        )
    )

    mild_probability = tf.reduce_sum(tf.gather(y_pred, MILD_NEGATIVE, axis=1), axis=1)
    penalty = penalty + tf.cast(outside_mild_penalty, y_pred.dtype) * masked(
        tf.logical_not(in_soft), mild_probability
    )
    penalty = penalty + tf.cast(neutral_mild_penalty, y_pred.dtype) * masked(
        is_neutral, mild_probability
    )
    margin = tf.cast(pairwise_margin, y_pred.dtype)
    happy_anger_margin = tf.nn.relu(
        tf.math.log(tf.clip_by_value(y_pred[:, ANGER], 1e-7, 1.0))
        - tf.math.log(tf.clip_by_value(y_pred[:, HAPPY], 1e-7, 1.0))
        + margin
    )
    neutral_mild_margin = tf.nn.relu(
        tf.math.log(tf.clip_by_value(mild_probability, 1e-7, 1.0))
        - tf.math.log(tf.clip_by_value(y_pred[:, NEUTRAL], 1e-7, 1.0))
        + margin
    )
    penalty = penalty + tf.cast(happy_anger_margin_weight, y_pred.dtype) * masked(
        is_happy, happy_anger_margin
    )
    penalty = penalty + tf.cast(neutral_mild_margin_weight, y_pred.dtype) * masked(
        is_neutral, neutral_mild_margin
    )
    return base_loss + penalty


def distillation_loss(
    y_true,
    y_pred,
    teacher_probs,
    temperature=2.0,
    distill_weight=0.5,
    group_weight=0.2,
    outside_mild_penalty=0.04,
    business_probs=None,
):
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    temperature = tf.cast(temperature, y_pred.dtype)
    eps = tf.cast(1e-7, y_pred.dtype)

    teacher_soft = tf.nn.softmax(tf.math.log(tf.clip_by_value(teacher_probs, eps, 1.0)) / temperature)
    student_soft = tf.nn.softmax(tf.math.log(tf.clip_by_value(y_pred, eps, 1.0)) / temperature)
    kd = tf.reduce_sum(
        teacher_soft * (tf.math.log(teacher_soft + eps) - tf.math.log(student_soft + eps)),
        axis=1,
    ) * tf.square(temperature)

    if business_probs is None:
        group_matrix = tf.one_hot(BUSINESS_GROUP, 5, dtype=y_pred.dtype)
        student_groups = tf.linalg.matmul(y_pred, group_matrix)
    else:
        student_groups = business_probs
    true_groups = tf.gather(BUSINESS_GROUP, y_true)
    group_loss = keras.losses.sparse_categorical_crossentropy(true_groups, student_groups)

    hard = business_aware_loss(
        y_true,
        y_pred,
        outside_mild_penalty=outside_mild_penalty,
    )
    distill_weight = tf.cast(distill_weight, y_pred.dtype)
    group_weight = tf.cast(group_weight, y_pred.dtype)
    return (1.0 - distill_weight) * hard + distill_weight * kd + group_weight * group_loss


@keras.utils.register_keras_serializable(package="emotion")
class BusinessAwareLoss(keras.losses.Loss):
    def __init__(
        self,
        group_keep=GROUP_KEEP,
        group_share=GROUP_SHARE,
        outside_mild_penalty=0.0,
        happy_anger_penalty=0.35,
        happy_anger_power=1.0,
        mild_neutral_penalty=0.18,
        neutral_mild_penalty=0.0,
        happy_anger_margin_weight=0.0,
        neutral_mild_margin_weight=0.0,
        pairwise_margin=0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.group_keep = float(group_keep)
        self.group_share = float(group_share)
        self.outside_mild_penalty = float(outside_mild_penalty)
        self.happy_anger_penalty = float(happy_anger_penalty)
        self.happy_anger_power = float(happy_anger_power)
        self.mild_neutral_penalty = float(mild_neutral_penalty)
        self.neutral_mild_penalty = float(neutral_mild_penalty)
        self.happy_anger_margin_weight = float(happy_anger_margin_weight)
        self.neutral_mild_margin_weight = float(neutral_mild_margin_weight)
        self.pairwise_margin = float(pairwise_margin)

    def call(self, y_true, y_pred):
        return business_aware_loss(
            y_true,
            y_pred,
            group_keep=self.group_keep,
            group_share=self.group_share,
            outside_mild_penalty=self.outside_mild_penalty,
            happy_anger_penalty=self.happy_anger_penalty,
            happy_anger_power=self.happy_anger_power,
            mild_neutral_penalty=self.mild_neutral_penalty,
            neutral_mild_penalty=self.neutral_mild_penalty,
            happy_anger_margin_weight=self.happy_anger_margin_weight,
            neutral_mild_margin_weight=self.neutral_mild_margin_weight,
            pairwise_margin=self.pairwise_margin,
        )

    def get_config(self):
        return {
            **super().get_config(),
            "group_keep": self.group_keep,
            "group_share": self.group_share,
            "outside_mild_penalty": self.outside_mild_penalty,
            "happy_anger_penalty": self.happy_anger_penalty,
            "happy_anger_power": self.happy_anger_power,
            "mild_neutral_penalty": self.mild_neutral_penalty,
            "neutral_mild_penalty": self.neutral_mild_penalty,
            "happy_anger_margin_weight": self.happy_anger_margin_weight,
            "neutral_mild_margin_weight": self.neutral_mild_margin_weight,
            "pairwise_margin": self.pairwise_margin,
        }
