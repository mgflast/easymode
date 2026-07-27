"""unet-membrain-groupnorm, but trained with SGD instead of Adam.

Identical forward architecture, loss, and metrics to unet-membrain-groupnorm
(GroupNorm(8), has-foreground dice). The ONLY difference is the optimizer:
high-momentum Nesterov SGD, the nnU-Net recipe for 3D segmentation, which the
original online models used before the switch to Adam. GroupNorm is kept (not
BatchNorm) so this is a single-variable change vs. the current default and is
immune to small-batch / per-replica normalization noise.

LR handling follows nnU-Net rather than the easymode CLI: this module exposes
INITIAL_LR and an lr_schedule(epoch, epochs) implementing nnU-Net's polynomial
decay (lr = initial * (1 - epoch/epochs) ** 0.9). segmentation/train.py detects
lr_schedule and uses it INSTEAD of the CLI lr_start/lr_end ramp, so the --lr
argument does not apply to this arch. Weight decay 3e-5, momentum 0.99, Nesterov
reproduce the nnU-Net SGD recipe.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model

INPUT_SHAPE = (160, 160, 160, 1)

# nnU-Net SGD recipe: initial LR 1e-2 with polynomial decay (power 0.9).
INITIAL_LR = 1e-2
LR_DECAY_POWER = 0.9
WEIGHT_DECAY = 3e-5


def lr_schedule(epoch, epochs):
    """nnU-Net PolyLR: lr = initial * (1 - epoch/epochs) ** 0.9."""
    frac = min(max(epoch / float(epochs), 0.0), 1.0)
    return float(INITIAL_LR * (1.0 - frac) ** LR_DECAY_POWER)


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

    has_fg = tf.reduce_sum(y_true_masked, axis=spatial_axes) > 0
    has_fg = tf.reshape(has_fg, [-1])
    per_sample_loss = tf.reshape(per_sample_loss, [-1])
    fg_losses = tf.boolean_mask(per_sample_loss, has_fg)
    return tf.cond(tf.size(fg_losses) > 0,
                   lambda: tf.reduce_mean(fg_losses),
                   lambda: tf.constant(0.0))

def masked_dice(y_true, y_pred):
    return 1.0 - masked_dice_loss(y_true, y_pred)

def combined_masked_bce_dice_loss(y_true, y_pred):
    return 0.3 * masked_bce_loss(y_true, y_pred) + 0.7 * masked_dice_loss(y_true, y_pred)


class ResBlock3D(layers.Layer):
    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

        self.conv1 = layers.Conv3D(filters, 3, padding='same', use_bias=False)
        self.bn1 = layers.GroupNormalization(groups=8)
        self.relu1 = layers.ReLU()

        self.conv2 = layers.Conv3D(filters, 3, padding='same', use_bias=False)
        self.bn2 = layers.GroupNormalization(groups=8)

        self.skip_conv = None
        self.skip_bn = None

    def build(self, input_shape):
        super().build(input_shape)
        if input_shape[-1] != self.filters:
            self.skip_conv = layers.Conv3D(self.filters, 1, padding='same', use_bias=False)
            self.skip_bn = layers.GroupNormalization(groups=8)

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        skip = inputs
        if self.skip_conv is not None:
            skip = self.skip_conv(inputs)
            skip = self.skip_bn(skip, training=training)

        x = x + skip
        return tf.nn.relu(x)


class EncoderBlock(layers.Layer):

    def __init__(self, filters: int, stride: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.stride = stride

        if stride > 1:
            self.downsample = layers.Conv3D(
                filters, kernel_size=3, strides=stride,
                padding='same', use_bias=False
            )
            self.downsample_bn = layers.GroupNormalization(groups=8)
            self.downsample_relu = layers.ReLU()
        else:
            self.downsample = None

        self.res_block = ResBlock3D(filters)

    def call(self, inputs, training=None):
        x = inputs

        if self.downsample is not None:
            x = self.downsample(x)
            x = self.downsample_bn(x, training=training)
            x = self.downsample_relu(x)

        x = self.res_block(x, training=training)
        return x


class DecoderBlock(layers.Layer):

    def __init__(self, filters: int, upsample_kernel_size: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.upsample_kernel_size = upsample_kernel_size

        if upsample_kernel_size > 1:
            self.upsample = layers.Conv3DTranspose(
                filters, kernel_size=upsample_kernel_size,
                strides=upsample_kernel_size, padding='same', use_bias=False
            )
            self.upsample_bn = layers.GroupNormalization(groups=8)
            self.upsample_relu = layers.ReLU()
        else:
            self.upsample = None

        self.res_block = ResBlock3D(filters)

    def call(self, inputs, skip_connection=None, training=None):
        x = inputs

        if self.upsample is not None:
            x = self.upsample(x)
            x = self.upsample_bn(x, training=training)
            x = self.upsample_relu(x)

        if skip_connection is not None:
            x = tf.concat([x, skip_connection], axis=-1)

        x = self.res_block(x, training=training)
        return x


class UNet(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        filters = [32, 64, 128, 256, 512, 1024]
        strides = [1, 2, 2, 2, 2, 2]
        upsample_kernel_sizes = [1, 2, 2, 2, 2, 2]

        self.encoders = []
        for i, (f, s) in enumerate(zip(filters, strides)):
            self.encoders.append(EncoderBlock(f, stride=s, name=f'encoder_{i}'))

        self.decoders = []
        decoder_filters = filters[:-1][::-1]
        decoder_upsample = upsample_kernel_sizes[1:][::-1]

        for i, (f, us) in enumerate(zip(decoder_filters, decoder_upsample)):
            self.decoders.append(DecoderBlock(f, upsample_kernel_size=us, name=f'decoder_{i}'))

        self.final_conv = layers.Conv3D(1, 1, activation='sigmoid', name='output')

        self.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=INITIAL_LR, momentum=0.99, nesterov=True, weight_decay=WEIGHT_DECAY),
            loss=combined_masked_bce_dice_loss,
            metrics=[masked_precision, masked_recall, masked_bce_loss, masked_dice],
            run_eagerly=False
        )

    def call(self, inputs, training=None):
        encoder_outputs = []
        x = inputs

        for encoder in self.encoders:
            x = encoder(x, training=training)
            encoder_outputs.append(x)

        skip_connections = encoder_outputs[:-1][::-1]
        x = encoder_outputs[-1]

        for i, decoder in enumerate(self.decoders):
            skip = skip_connections[i] if i < len(skip_connections) else None
            x = decoder(x, skip_connection=skip, training=training)

        output = self.final_conv(x)
        return output


def create():
    return UNet()
