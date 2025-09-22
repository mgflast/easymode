import tensorflow as tf
from tensorflow.keras import layers, Model
from easymode.ddw.loss import DDWLoss

class ResBlock3D(layers.Layer):
    def __init__(self, filters: int, drop_prob: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

        self.conv1 = layers.Conv3D(filters // 2, 3, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(drop_prob)
        self.relu1 = layers.LeakyReLU(alpha=0.05)

        self.conv2 = layers.Conv3D(filters // 2, 3, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(drop_prob)
        self.relu2 = layers.LeakyReLU(alpha=0.05)

        self.conv3 = layers.Conv3D(filters, 3, padding='same', use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.dropout3 = layers.Dropout(drop_prob)
        self.relu3 = layers.LeakyReLU(alpha=0.05)

        self.skip_conv = None
        self.skip_bn = None

    def build(self, input_shape):
        super().build(input_shape)
        if input_shape[-1] != self.filters:
            self.skip_conv = layers.Conv3D(self.filters, 1, padding='same', use_bias=False)
            self.skip_bn = layers.BatchNormalization()

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.dropout2(x, training=training)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.dropout3(x, training=training)
        x = self.relu3(x)

        skip = inputs
        if self.skip_conv is not None:
            skip = self.skip_conv(inputs)
            skip = self.skip_bn(skip, training=training)

        return x + skip


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

        filters = [32, 64, 128, 256, 512]
        strides = [1, 2, 2, 2, 2]
        upsample_kernel_sizes = [1, 2, 2, 2, 2]

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
            loss=DDWLoss,
            metrics=['mae'],
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

