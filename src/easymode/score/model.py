import tensorflow as tf
from tensorflow.keras import layers, Model

PATCH = 96
EPS = 1e-5  # torch GroupNorm / LayerNorm eps; Keras default is 1e-3
_CONV = [(8, 1), (16, 2), (32, 2), (64, 2), (64, 2), (64, 2)]  # (filters, stride)


def create():
    inp = layers.Input(shape=(None, None, None, 1), name='patch')
    x = inp
    for i, (filters, stride) in enumerate(_CONV):
        if stride == 1:
            x = layers.Conv3D(filters, 3, strides=1, padding='same', name=f'conv{i}')(x)
        else:
            # torch Conv3d(stride=2, padding=1); Keras 'same' pads asymmetrically and shifts the grid
            x = layers.ZeroPadding3D(1, name=f'pad{i}')(x)
            x = layers.Conv3D(filters, 3, strides=2, padding='valid', name=f'conv{i}')(x)
        x = layers.GroupNormalization(groups=8, epsilon=EPS, name=f'gn{i}')(x)
        x = layers.Activation(tf.nn.silu, name=f'silu{i}')(x)
    x = layers.GlobalAveragePooling3D(name='gap')(x)
    x = layers.Dense(16, name='features')(x)
    x = layers.LayerNormalization(epsilon=EPS, name='features_ln')(x)
    x = layers.Activation(tf.nn.silu, name='features_silu')(x)
    score = layers.Dense(1, use_bias=False, name='score_head')(x)
    log_precision = layers.Dense(1, name='log_precision_head')(x)
    return Model(inp, [score, log_precision], name='misalignment')
