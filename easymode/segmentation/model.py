import tensorflow as tf
from tensorflow.keras import layers, Model

def masked_bce_loss(y_true, y_pred, fn_weight=3.0):
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

def masked_accuracy(y_true, y_pred):
    ignore = tf.equal(y_true, 2.0)
    y_true_bin = tf.where(ignore, 0.0, y_true)
    y_pred_bin = tf.cast(y_pred > 0.5, y_true_bin.dtype)

    valid = tf.logical_not(ignore)
    correct = tf.equal(y_true_bin, y_pred_bin)
    correct = tf.logical_and(correct, valid)

    correct = tf.cast(correct, y_pred.dtype)
    denom = tf.reduce_sum(tf.cast(valid, y_pred.dtype))
    return tf.reduce_sum(correct) / tf.maximum(denom, 1.0)

def masked_dice_loss(y_true, y_pred, smooth=1e-6):
    mask = tf.cast(y_true != 2, tf.float32)
    y_true_masked = tf.reshape(y_true * mask, [-1])
    y_pred_masked = tf.reshape(y_pred * mask, [-1])

    intersection = tf.reduce_sum(y_true_masked * y_pred_masked)
    union = tf.reduce_sum(y_true_masked) + tf.reduce_sum(y_pred_masked)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice

# def masked_dice(y_true, y_pred):
#     return 1.0 - masked_dice_loss(y_true, y_pred)

def combined_masked_bce_dice_loss(y_true, y_pred):
    return 1.0 * masked_bce_loss(y_true, y_pred) #+  0.7 * masked_dice_loss(y_true, y_pred)


class ResBlock3D(layers.Layer):
    """3D Residual block with batch normalization and ReLU activation."""

    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

        # First conv layer
        self.conv1 = layers.Conv3D(filters, 3, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

        # Second conv layer
        self.conv2 = layers.Conv3D(filters, 3, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()

        # Skip connection adjustment if needed
        self.skip_conv = None
        self.skip_bn = None

    def build(self, input_shape):
        super().build(input_shape)
        # Add skip connection conv if input channels != output channels
        if input_shape[-1] != self.filters:
            self.skip_conv = layers.Conv3D(self.filters, 1, padding='same', use_bias=False)
            self.skip_bn = layers.BatchNormalization()

    def call(self, inputs, training=None):
        # Main path
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        # Skip connection
        skip = inputs
        if self.skip_conv is not None:
            skip = self.skip_conv(inputs)
            skip = self.skip_bn(skip, training=training)

        # Add and activate
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
            self.downsample_bn = layers.BatchNormalization()
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
            self.upsample_bn = layers.BatchNormalization()
            self.upsample_relu = layers.ReLU()
        else:
            self.upsample = None

        self.res_block = ResBlock3D(filters)

    def call(self, inputs, skip_connection=None, training=None):
        x = inputs

        # Upsample if needed
        if self.upsample is not None:
            x = self.upsample(x)
            x = self.upsample_bn(x, training=training)
            x = self.upsample_relu(x)

        # Concatenate with skip connection
        if skip_connection is not None:
            x = tf.concat([x, skip_connection], axis=-1)

        # Apply residual block
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
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=combined_masked_bce_dice_loss,
            metrics=[masked_accuracy],
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

        # Final output
        output = self.final_conv(x)
        return output


def create():
    model = UNet()
    return model

