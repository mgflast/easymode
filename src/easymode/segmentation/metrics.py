"""Shared streaming segmentation metrics (dataset-level, threshold 0.5).

segmentation/train.py compiles these for EVERY architecture. Unlike the per-voxel
masked_precision/masked_recall functions the arch modules define (which Keras wraps
in MeanMetricWrapper and averages per batch — scoring a correctly-silent background
patch as a phantom 0 and structurally unable to reach 1.0), these accumulate TP/FP/FN
across the whole epoch and divide once, so the reported value is true dataset-level
precision/recall. Label==2 (ignore / out-of-bounds) voxels are excluded; false
positives on background patches still count.
"""
import tensorflow as tf


class MaskedPrecision(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, name='masked_precision', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        valid = tf.cast(tf.not_equal(y_true, 2.0), tf.float32)
        y_true_pos = tf.cast(tf.equal(y_true, 1.0), tf.float32) * valid
        y_pred_pos = tf.cast(y_pred > self.threshold, tf.float32) * valid
        self.tp.assign_add(tf.reduce_sum(y_pred_pos * y_true_pos))
        self.fp.assign_add(tf.reduce_sum(y_pred_pos * (1.0 - y_true_pos)))

    def result(self):
        return tf.math.divide_no_nan(self.tp, self.tp + self.fp)

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)


class MaskedRecall(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, name='masked_recall', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        valid = tf.cast(tf.not_equal(y_true, 2.0), tf.float32)
        y_true_pos = tf.cast(tf.equal(y_true, 1.0), tf.float32) * valid
        y_pred_pos = tf.cast(y_pred > self.threshold, tf.float32) * valid
        self.tp.assign_add(tf.reduce_sum(y_pred_pos * y_true_pos))
        self.fn.assign_add(tf.reduce_sum((1.0 - y_pred_pos) * y_true_pos))

    def result(self):
        return tf.math.divide_no_nan(self.tp, self.tp + self.fn)

    def reset_state(self):
        self.tp.assign(0.0)
        self.fn.assign(0.0)


def streaming_precision_recall():
    """Fresh streaming P/R metric instances named to match the training_log.csv columns."""
    return [MaskedPrecision(name='masked_precision'), MaskedRecall(name='masked_recall')]
