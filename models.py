import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Concatenate, Conv2D, Input,
                                     LeakyReLU, MaxPooling2D, UpSampling2D, concatenate)
from tensorflow.keras.models import Model

import config


def conv_block(x, filters):
    x = Conv2D(filters, (3, 3), activation="relu", padding="same")(x)
    return Conv2D(filters, (3, 3), activation="relu", padding="same")(x)


def up_block(x, skip, filters):
    x = concatenate([UpSampling2D(size=(2, 2))(x), skip], axis=-1)
    return conv_block(x, filters)


def unet(input_shape=config.INPUT_SHAPE):
    """U-Net with sigmoid output in [0, 1]. Also used as the Pix2Pix generator."""
    inputs = Input(shape=input_shape)

    c1 = conv_block(inputs, 64)
    c2 = conv_block(MaxPooling2D((2, 2))(c1), 128)
    c3 = conv_block(MaxPooling2D((2, 2))(c2), 256)
    c4 = conv_block(MaxPooling2D((2, 2))(c3), 512)
    c5 = conv_block(MaxPooling2D((2, 2))(c4), 1024)

    x = up_block(c5, c4, 512)
    x = up_block(x, c3, 256)
    x = up_block(x, c2, 128)
    x = up_block(x, c1, 64)

    outputs = Conv2D(3, (1, 1), activation="sigmoid")(x)
    return Model(inputs=[inputs], outputs=[outputs])


def discriminator(input_shape=config.INPUT_SHAPE):
    """PatchGAN (70x70 receptive field). Outputs a 32x32 map of logits, one per patch."""
    hazy = Input(shape=input_shape, name="disc_input")
    clear = Input(shape=input_shape, name="disc_target")
    x = Concatenate()([hazy, clear])

    # (filters, stride, batch norm); no BN on the first layer, as in the pix2pix paper
    for filters, stride, use_bn in [(64, 2, False), (128, 2, True), (256, 2, True), (512, 1, True)]:
        x = Conv2D(filters, (4, 4), strides=(stride, stride), padding="same", use_bias=not use_bn)(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = LeakyReLU(negative_slope=0.2)(x)

    x = Conv2D(1, (4, 4), strides=(1, 1), padding="same", name="disc_output")(x)
    return Model(inputs=[hazy, clear], outputs=x, name="patchgan_discriminator")


bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(disc_fake, gen_output, target, lambda_l1=100):
    gan_loss = bce(tf.ones_like(disc_fake), disc_fake)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    return gan_loss + lambda_l1 * l1_loss, gan_loss, l1_loss


def discriminator_loss(disc_real, disc_fake):
    return bce(tf.ones_like(disc_real), disc_real) + bce(tf.zeros_like(disc_fake), disc_fake)
