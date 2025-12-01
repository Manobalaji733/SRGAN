# src/discriminator.py - example discriminator (logits output)
import tensorflow as tf
from tensorflow.keras import layers, Model, Input

def conv_block(x, filters, kernel_size=3, strides=1, use_bn=True):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    return x

def build_discriminator(input_shape=(96,96,3)):
    inp = Input(shape=input_shape)
    x = layers.Conv2D(64, 3, strides=1, padding='same')(inp)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = conv_block(x, 64, strides=2)
    x = conv_block(x, 128, strides=1)
    x = conv_block(x, 128, strides=2)
    x = conv_block(x, 256, strides=1)
    x = conv_block(x, 256, strides=2)
    x = conv_block(x, 512, strides=1)
    x = conv_block(x, 512, strides=2)

    x = layers.Flatten()(x)
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # NOTE: logits output (no sigmoid) to match losses with from_logits=True
    out = layers.Dense(1)(x)

    return Model(inputs=inp, outputs=out, name="discriminator")
