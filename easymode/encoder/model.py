import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Reshape, Conv3D, Conv3DTranspose, GlobalAveragePooling3D, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

INPUT_SHAPE = (256, 256, 256, 1)
LATENT_DIMENSIONALITY = 128


def encoder_block(x, f):
    x = Conv3D(filters=f, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.nn.swish(x)
    x = Conv3D(filters=f, kernel_size=3, strides=2, padding='same')(x)  # downsample /2
    x = BatchNormalization()(x)
    x = tf.nn.swish(x)
    return x


def decoder_block(x, f):
    x = Conv3DTranspose(filters=f, kernel_size=3, strides=2, padding='same')(x)  # upsample x2
    x = BatchNormalization()(x)
    x = tf.nn.swish(x)
    x = Conv3D(filters=f, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.nn.swish(x)
    return x


class Sampling(Layer):
    def call(self, inputs):
        mu, logvar = inputs
        eps = tf.random.normal(tf.shape(mu))
        return mu + tf.exp(0.5 * logvar) * eps


def encoder():
    inputs = Input(shape=INPUT_SHAPE)

    b1 = encoder_block(inputs, 32)    # 256 -> 128
    b2 = encoder_block(b1, 64)        # 128 -> 64
    b3 = encoder_block(b2, 128)       # 64  -> 32
    b4 = encoder_block(b3, 256)       # 32  -> 16
    b5 = encoder_block(b4, 512)       # 16  -> 8

    x = GlobalAveragePooling3D()(b5)  # (B, 512)

    # small MLP for latent
    x = Dense(512, activation='relu')(x)

    mu = Dense(LATENT_DIMENSIONALITY, name='mu')(x)
    logvar = Dense(LATENT_DIMENSIONALITY, name='logvar')(x)
    z = Sampling(name='z')([mu, logvar])

    return Model(inputs, [mu, logvar, z], name='encoder')


def decoder():
    z_in = Input(shape=(LATENT_DIMENSIONALITY,))

    # match encoder deepest spatial size: 8 x 8 x 8, 512 channels
    x = Dense(8 * 8 * 8 * 512, activation='relu')(z_in)
    x = Reshape((8, 8, 8, 512))(x)

    x = decoder_block(x, 256)  # 8   -> 16
    x = decoder_block(x, 128)  # 16  -> 32
    x = decoder_block(x, 64)   # 32  -> 64
    x = decoder_block(x, 32)   # 64  -> 128
    x = decoder_block(x, 16)   # 128 -> 256

    x_out = Conv3D(INPUT_SHAPE[-1], 3, activation='linear', padding='same', name='recon')(x)

    return Model(z_in, x_out, name='decoder')


class VAE(Model):
    def __init__(self, encoder, decoder, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta

        self.total_loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.recon_loss_tracker = tf.keras.metrics.Mean(name='recon_loss')
        self.kl_loss_tracker = tf.keras.metrics.Mean(name='kl_loss')

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def call(self, inputs, training=False):
        mu, logvar, z = self.encoder(inputs, training=training)
        recon = self.decoder(z, training=training)
        return recon

    def train_step(self, data):
        if isinstance(data, (tuple, list)):
            x = data[0]
        else:
            x = data

        with tf.GradientTape() as tape:
            mu, logvar, z = self.encoder(x, training=True)
            recon = self.decoder(z, training=True)

            recon_error = x - recon
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(recon_error), axis=[1, 2, 3, 4])
            )

            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1.0 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
            )

            loss = recon_loss + self.beta * kl_loss

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_loss_tracker.update_state(loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            'loss': self.total_loss_tracker.result(),
            'recon_loss': self.recon_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
        }


def create_vae(beta=0.1, lr=1e-4):
    enc = encoder()
    dec = decoder()
    vae = VAE(enc, dec, beta=beta)
    vae.compile(optimizer=Adam(lr))
    return vae
