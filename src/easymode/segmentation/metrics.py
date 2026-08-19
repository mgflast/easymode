"""Streaming, dataset-level soft precision/recall (no threshold).

The raw probability is the fractional TP/FP/FN mass, so the metrics move from
epoch 1 even while no prediction crosses 0.5, and converge to the hard-threshold
values as the model becomes confident. Unlike the per-batch masked_precision/
masked_recall in losses.py (which Keras averages per batch, scoring a correctly-
silent background patch as a phantom 0), TP/FP/FN accumulate across the whole
epoch and divide once. Label==2 (ignore / out-of-bounds) voxels are excluded;
false positives on background patches still count.
"""
import tensorflow as tf


class SoftMaskedPrecision(tf.keras.metrics.Metric):
    def __init__(self, name='precision', **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        valid = tf.cast(tf.not_equal(y_true, 2.0), tf.float32)
        y_true_pos = tf.cast(tf.equal(y_true, 1.0), tf.float32) * valid
        y_pred_pos = tf.cast(y_pred, tf.float32) * valid
        self.tp.assign_add(tf.reduce_sum(y_pred_pos * y_true_pos))
        self.fp.assign_add(tf.reduce_sum(y_pred_pos * (1.0 - y_true_pos)))

    def result(self):
        return tf.math.divide_no_nan(self.tp, self.tp + self.fp)

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)


class SoftMaskedRecall(tf.keras.metrics.Metric):
    def __init__(self, name='recall', **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        valid = tf.cast(tf.not_equal(y_true, 2.0), tf.float32)
        y_true_pos = tf.cast(tf.equal(y_true, 1.0), tf.float32) * valid
        y_pred_pos = tf.cast(y_pred, tf.float32) * valid
        self.tp.assign_add(tf.reduce_sum(y_pred_pos * y_true_pos))
        self.fn.assign_add(tf.reduce_sum((1.0 - y_pred_pos) * y_true_pos))

    def result(self):
        return tf.math.divide_no_nan(self.tp, self.tp + self.fn)

    def reset_state(self):
        self.tp.assign(0.0)
        self.fn.assign(0.0)
