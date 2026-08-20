"""unet-membrain-groupnorm with one downsampling level fewer.

Identical blocks, filters, losses and optimizer as unet-membrain-groupnorm; only the deepest
(1024-filter) level is dropped. Stride product 16 instead of 32, receptive field 219 instead
of 443 voxels, and ~32M instead of ~129M parameters -- the removed level held most of them.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model

from easymode.segmentation.losses import (combined_masked_bce_dice_loss, masked_bce_loss,
                                          masked_dice, masked_dice_loss, masked_precision,
                                          masked_recall)
from easymode.segmentation.models.unet_membrain_groupnorm import EncoderBlock, DecoderBlock

INPUT_SHAPE = (160, 160, 160, 1)


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

        self.final_conv = layers.Conv3D(1, 1, activation='sigmoid', name='output')

        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
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

        return self.final_conv(x)


def create():
    return UNet()
