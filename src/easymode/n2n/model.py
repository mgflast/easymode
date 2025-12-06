import tensorflow as tf
from tensorflow.keras import layers, Model


def gaussian_kernel_3d(size=11, sigma=1.5):
    ax = tf.range(-size // 2 + 1., size // 2 + 1., dtype=tf.float32)
    zz, yy, xx = tf.meshgrid(ax, ax, ax, indexing='ij')
    kernel = tf.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2 * sigma ** 2))
    kernel = kernel / tf.reduce_sum(kernel)
    return kernel[:, :, :, None, None]


def ssim_3d(img1, img2, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
    kernel = gaussian_kernel_3d(filter_size, filter_sigma)

    def conv3d(img, kernel):
        return tf.nn.conv3d(img, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')

    mu1 = conv3d(img1, kernel)
    mu2 = conv3d(img2, kernel)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv3d(img1 * img1, kernel) - mu1_sq
    sigma2_sq = conv3d(img2 * img2, kernel) - mu2_sq
    sigma12 = conv3d(img1 * img2, kernel) - mu1_mu2

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return tf.reduce_mean(ssim_map)


def masked_l1_ssim_loss(y_true, y_pred, alpha=0.8, edge=16, max_val=1.0):
    shape = tf.shape(y_true)
    d, h, w = shape[1], shape[2], shape[3]

    mask_d = tf.logical_and(tf.range(d) >= edge, tf.range(d) < d - edge)
    mask_h = tf.logical_and(tf.range(h) >= edge, tf.range(h) < h - edge)
    mask_w = tf.logical_and(tf.range(w) >= edge, tf.range(w) < w - edge)

    mask = tf.logical_and(tf.logical_and(
        mask_d[None, :, None, None, None],
        mask_h[None, None, :, None, None]),
        mask_w[None, None, None, :, None])

    mask_float = tf.cast(mask, y_pred.dtype)

    per_voxel = tf.abs(y_true - y_pred)
    masked_per_voxel = per_voxel * mask_float
    denom = tf.reduce_sum(mask_float)
    l1_loss = tf.reduce_sum(masked_per_voxel) / tf.maximum(denom, 1.0)

    ssim_val = ssim_3d(y_true, y_pred, max_val=max_val)
    ssim_loss = 1.0 - ssim_val

    return alpha * l1_loss + (1.0 - alpha) * ssim_loss

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

        filters = [32, 64, 128, 256]
        strides = [1, 2, 2, 2]
        upsample_kernel_sizes = [1, 2, 2, 2]

        self.encoders = []
        for i, (f, s) in enumerate(zip(filters, strides)):
            self.encoders.append(EncoderBlock(f, stride=s, name=f'encoder_{i}'))

        self.decoders = []
        decoder_filters = filters[:-1][::-1]
        decoder_upsample = upsample_kernel_sizes[1:][::-1]

        for i, (f, us) in enumerate(zip(decoder_filters, decoder_upsample)):
            self.decoders.append(DecoderBlock(f, upsample_kernel_size=us, name=f'decoder_{i}'))

        self.final_conv = layers.Conv3D(1, 1, activation='linear', name='output', use_bias=False)

        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0004),
            loss=masked_l1_ssim_loss,
            metrics=['mae', 'mse'],
            run_eagerly=False,
        )


    def call(self, inputs, training=None):
        # Encoder path
        encoder_outputs = []
        x = inputs

        for encoder in self.encoders:
            x = encoder(x, training=training)
            encoder_outputs.append(x)

        # Decoder path
        skip_connections = encoder_outputs[:-1][::-1]  # Reverse, exclude bottleneck
        x = encoder_outputs[-1]  # Start with bottleneck

        for i, decoder in enumerate(self.decoders):
            skip = skip_connections[i] if i < len(skip_connections) else None
            x = decoder(x, skip_connection=skip, training=training)

        # Final output
        residual = self.final_conv(x)
        return inputs - residual


def create():
    model = UNet()
    return model

