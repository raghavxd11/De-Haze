import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Concatenate,
)
from tensorflow.keras.models import Model
from config import config
from unet_model import unet


def generator_model(input_shape=config.input_shape):
    return unet(input_shape)


def generator_loss(disc_generated_output, gen_output, target, LAMBDA=100):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # GAN loss
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # Mean absolute error (L1 loss)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    # Total generator loss
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


def discriminator_model(input_shape=config.input_shape):
    """PatchGAN discriminator (70×70 receptive field).

    Outputs a spatial map of shape (None, H', W', 1) where each element
    represents a local patch verdict. BinaryCrossentropy(from_logits=True)
    is applied element-wise during training, identical to the pix2pix paper.
    """
    input_image = Input(shape=input_shape, name="disc_input")
    target_image = Input(shape=input_shape, name="disc_target")

    # Concatenate input_image and target_image along the channel dimension
    x = Concatenate()([input_image, target_image])          # (B,H,W,6)

    # Block 1 – no BN on first layer (pix2pix convention)
    x = Conv2D(64,  (4, 4), strides=(2, 2), padding="same",
               use_bias=True)(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)   # 128×128

    # Block 2
    x = Conv2D(128, (4, 4), strides=(2, 2), padding="same",
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)   # 64×64

    # Block 3
    x = Conv2D(256, (4, 4), strides=(2, 2), padding="same",
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)   # 32×32

    # Block 4 – stride-1 to keep spatial resolution before final conv
    x = Conv2D(512, (4, 4), strides=(1, 1), padding="same",
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)   # 32×32

    # Patch output – 1 channel, no activation (logits for BCE)
    x = Conv2D(1,   (4, 4), strides=(1, 1), padding="same",
               use_bias=True, name="disc_output")(x)        # (B,32,32,1)

    model = Model(inputs=[input_image, target_image], outputs=x,
                  name="patchgan_discriminator")
    return model


def discriminator_loss(disc_real_output, disc_generated_output):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # Real loss
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    # Generated loss
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    # Total discriminator loss
    total_disc_loss = real_loss + generated_loss

    return total_disc_loss
