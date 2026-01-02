import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, ReLU, GlobalAveragePooling2D, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def make_encoder():
    input = Input(shape=(None, None, 1))

    block1_0 = Conv2D(32, 3, strides=1, activation='relu', padding='same')(input)
    block1_1 = Conv2D(32, 3, strides=2, activation='relu', padding='same')(block1_0)

    block2_0 = Conv2D(64, 3, strides=1, activation='relu', padding='same')(block1_1)
    block2_1 = Conv2D(64, 3, strides=2, activation='relu', padding='same')(block2_0)

    block3_0 = Conv2D(128, 3, strides=1, activation='relu', padding='same')(block2_1)
    block3_1 = Conv2D(128, 3, strides=2, activation='relu', padding='same')(block3_0)

    block4_0 = Conv2D(256, 3, strides=1, activation='relu', padding='same')(block3_1)
    block4_1 = Conv2D(256, 3, strides=2, activation='relu', padding='same')(block4_0)

    block5_0 = Conv2D(512, 3, strides=1, activation='relu', padding='same')(block4_1)
    output = GlobalAveragePooling2D()(block4_0)

    return Model(input, output, name="encoder")


def create():
    x0 = Input(shape=(None, None, 1), name="x0")
    x1 = Input(shape=(None, None, 1), name="x1")

    encoder = make_encoder()

    e0 = encoder(x0)
    e1 = encoder(x1)

    z = Concatenate()([e0, e1, tf.abs(e0 - e1), e0 * e1])
    z = Dense(256, activation='relu')(z)
    z = Dense(128, activation='relu')(z)

    out = Dense(1, activation='sigmoid')(z)

    model = Model(inputs=[x0, x1], outputs=out)

    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])

    return model