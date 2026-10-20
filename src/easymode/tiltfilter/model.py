import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, ReLU, GlobalAveragePooling2D, Dense, Concatenate, BatchNormalization, Lambda, Multiply, Subtract
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def block(x, f):
    x = Conv2D(filters=f, strides=1, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=f, strides=2, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def make_encoder():
    input = Input(shape=(None, None, 1))

    b1 = block(input, 32)
    b2 = block(b1, 64)
    b3 = block(b2, 128)
    b4 = block(b3, 256)
    b5 = block(b4, 512)

    output = GlobalAveragePooling2D()(b5)

    return Model(input, output, name="encoder")


def create():
    x0 = Input(shape=(None, None, 1), name="x0")
    x1 = Input(shape=(None, None, 1), name="x1")

    encoder = make_encoder()

    e0 = encoder(x0)
    e1 = encoder(x1)

    # layers rather than tf ops: Keras 3 cannot apply a tf op to a symbolic tensor. None has weights.
    z = Concatenate()([e0, e1, Lambda(tf.abs)(Subtract()([e0, e1])), Multiply()([e0, e1])])
    z = Dense(256, activation='relu')(z)
    z = Dense(64, activation='relu')(z)

    out = Dense(1, activation='sigmoid')(z)

    model = Model(inputs=[x0, x1], outputs=out)

    # jit_compile=False, as on Keras 2: Keras 3 would XLA-compile select_tilts' predict() for every new image size
    # (2-4 s each) to save 2% per call.
    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.Precision(name="prec"),
                           tf.keras.metrics.Recall(name="rec"),
                           'accuracy'],
                  jit_compile=False)

    return model