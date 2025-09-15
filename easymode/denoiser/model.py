import tensorflow as tf
from tensorflow.keras import layers, Model


def l1_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))


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


class VolumeDenoiserUNet(Model):
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

        self.final_conv = layers.Conv3D(1, 1, activation='linear', name='output')

        self.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.0001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            ),
            loss=l1_loss,
            metrics=['mae', 'mse'],
            run_eagerly=False,
            steps_per_execution=16,
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
        output = self.final_conv(x)
        return output

    def train_model(self, train_data, val_data, epochs=100, batch_size=1):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=15,
                restore_best_weights=True,
                monitor='val_loss',
                min_delta=1e-6
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                monitor='val_loss',
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_denoiser_model.h5',
                save_best_only=True,
                monitor='val_loss'
            )
        ]

        return self.fit(
            train_data.batch(batch_size),
            validation_data=val_data.batch(batch_size),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

    def predict_volume(self, noisy_volume):
        return self.predict(noisy_volume)


def create_denoiser():
    model = VolumeDenoiserUNet()
    return model


def prepare_denoising_dataset(full_volumes, denoised_volumes, validation_split=0.2):
    """Prepare dataset for denoising training"""
    dataset = tf.data.Dataset.from_tensor_slices((full_volumes, denoised_volumes))
    dataset = dataset.shuffle(len(full_volumes))

    val_size = int(len(full_volumes) * validation_split)
    train_data = dataset.skip(val_size)
    val_data = dataset.take(val_size)

    return train_data, val_data