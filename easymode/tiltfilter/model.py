import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, ReLU, GlobalAveragePooling2D, Dense, Concatenate, BatchNormalization
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

    z = Concatenate()([e0, e1, tf.abs(e0 - e1), e0 * e1])
    z = Dense(256, activation='relu')(z)
    z = Dense(64, activation='relu')(z)

    out = Dense(1, activation='sigmoid')(z)

    model = Model(inputs=[x0, x1], outputs=out)

    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.Precision(name="prec"),
                           tf.keras.metrics.Recall(name="rec"),
                           'accuracy'])

    return model