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

from emotion.config import ANGER, DISGUST, FEAR, HAPPY, NEUTRAL, SAD, SURPRISE

# 轻度负向三类：允许互相混淆
MILD_NEGATIVE = (FEAR, DISGUST, SAD)

# label smoothing 强度
SMOOTHING = 0.02
# 组内样本把多少置信度分给"兄弟类别"
GROUP_KEEP = 0.80
GROUP_SHARE = 0.09


def business_aware_loss(y_true, y_pred):
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
        onehot * tf.cast(GROUP_KEEP, y_pred.dtype)
        + peers * tf.cast(GROUP_SHARE, y_pred.dtype)
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
        0.35 * masked(is_happy, y_pred[:, ANGER])
        # happy 被判成 sad / neutral
        + 0.18 * masked(is_happy, y_pred[:, SAD] + y_pred[:, NEUTRAL])
        # anger 被判成 happy
        + 0.12 * masked(is_anger, y_pred[:, HAPPY])
        # 轻度负向被判成 neutral / happy / anger / surprise
        + 0.18 * masked(in_soft, y_pred[:, NEUTRAL])
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

    return base_loss + penalty
