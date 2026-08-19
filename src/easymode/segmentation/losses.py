"""Masked losses and per-voxel metrics shared by every segmentation architecture.

Label convention: 0 = background, 1 = foreground, 2 = ignore (out of bounds /
unannotated). Ignored voxels are excluded from every reduction here.

masked_dice_loss averages only over samples that contain foreground; samples
without any are dropped, and a batch (or replica) with no foreground at all
returns 0.0 -- which Keras reports as masked_dice == 1.0, so the logged dice is
inflated whenever background-only patches are in the batch.
"""
import tensorflow as tf


def masked_bce_loss(y_true, y_pred, fn_weight=1.0):
    ignore = tf.equal(y_true, 2.0)
    y_true_bin = tf.where(ignore, 0.0, y_true)

    eps = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

    per_voxel = -(
        fn_weight * y_true_bin * tf.math.log(y_pred) +
        (1.0 - y_true_bin) * tf.math.log(1.0 - y_pred)
    )

    mask = tf.cast(tf.logical_not(ignore), y_pred.dtype)
    per_voxel = per_voxel * mask

    denom = tf.reduce_sum(mask)
    return tf.reduce_sum(per_voxel) / tf.maximum(denom, 1.0)

def masked_precision(y_true, y_pred):
    ignore = tf.equal(y_true, 2.0)
    y_true_bin = tf.cast(tf.where(ignore, 0.0, y_true), tf.float32)
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    valid = tf.cast(tf.logical_not(ignore), tf.float32)

    tp = tf.reduce_sum(y_pred_bin * y_true_bin * valid)
    fp = tf.reduce_sum(y_pred_bin * (1.0 - y_true_bin) * valid)
    return tp / tf.maximum(tp + fp, 1.0)

def masked_recall(y_true, y_pred):
    ignore = tf.equal(y_true, 2.0)
    y_true_bin = tf.cast(tf.where(ignore, 0.0, y_true), tf.float32)
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    valid = tf.cast(tf.logical_not(ignore), tf.float32)

    tp = tf.reduce_sum(y_pred_bin * y_true_bin * valid)
    fn = tf.reduce_sum((1.0 - y_pred_bin) * y_true_bin * valid)
    return tp / tf.maximum(tp + fn, 1.0)

def masked_dice_loss(y_true, y_pred, smooth=1e-6):
    mask = tf.cast(y_true != 2, tf.float32)
    y_true_masked = y_true * mask
    y_pred_masked = y_pred * mask

    spatial_axes = list(range(1, len(y_true.shape)))
    intersection = tf.reduce_sum(y_true_masked * y_pred_masked, axis=spatial_axes)
    union = tf.reduce_sum(y_true_masked, axis=spatial_axes) + tf.reduce_sum(y_pred_masked, axis=spatial_axes)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    per_sample_loss = 1.0 - dice

    # mean over foreground-containing samples, written without boolean_mask/cond so the
    # graph has static shapes (XLA-compilable); identical values, incl. 0.0 for no-fg batches
    has_fg = tf.cast(tf.reduce_sum(y_true_masked, axis=spatial_axes) > 0, tf.float32)
    has_fg = tf.reshape(has_fg, [-1])
    per_sample_loss = tf.reshape(per_sample_loss, [-1])
    return tf.reduce_sum(per_sample_loss * has_fg) / tf.maximum(tf.reduce_sum(has_fg), 1.0)

def masked_dice(y_true, y_pred):
    return 1.0 - masked_dice_loss(y_true, y_pred)

def combined_masked_bce_dice_loss(y_true, y_pred):
    return 0.3 * masked_bce_loss(y_true, y_pred) + 0.7 * masked_dice_loss(y_true, y_pred)
