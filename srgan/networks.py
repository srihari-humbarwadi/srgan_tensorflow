import numpy as np
import tensorflow as tf

from srgan.layers import (ResidualBlock, UpscaleBlock, ConvBlock)


class Generator(tf.keras.Model):

    def __init__(self,
                 num_residual_blocks=16,
                 upscale_factor=4,
                 weights=None,
                 **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.upscale_factor = upscale_factor
        self.num_residual_blocks = num_residual_blocks

        self.conv_1 = tf.keras.layers.Conv2D(64, 9, 1, 'same')
        self.conv_2 = tf.keras.layers.Conv2D(64, 3, 1, 'same')
        self.conv_3 = tf.keras.layers.Conv2D(3, 9, 1, 'same')
        self.prelu = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.8)

        self.residual_blocks = tf.keras.Sequential(
            [ResidualBlock(64) for _ in range(num_residual_blocks)],
            name='ResidualBlocks')
        self.upsample_blocks = tf.keras.Sequential(
            [UpscaleBlock() for _ in range(int(np.log2(upscale_factor)))],
            name='UpscaleBlocks')

        if weights:
            try:
                print('restoring generator weights from: ', weights)
                self.load_weights(weights)
            except Exception:
                raise ValueError

    def call(self, image, training=False):
        image = image / 255.
        x = self.conv_1(image)
        x = self.prelu(x)
        skip = x
        x = self.residual_blocks(x, training=training)
        x = self.conv_2(x)
        x = self.bn(x, training=training)
        x = x + skip
        x = self.upsample_blocks(x)
        x = self.conv_3(x)
        sr_image = tf.nn.tanh(x)
        return sr_image


class Discriminator(tf.keras.Model):

    def __init__(self, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.conv_1 = tf.keras.layers.Conv2D(64, 3, 1, 'same')
        self.conv_blocks = tf.keras.Sequential(name='ConvBlocks')
        self.conv_blocks.add(ConvBlock(64, 2))
        for i in range(1, 4):
            self.conv_blocks.add(ConvBlock(64 * 2**i, 1))
            self.conv_blocks.add(ConvBlock(64 * 2**i, 2))
        self.fc_1 = tf.keras.layers.Dense(1024)
        self.fc_2 = tf.keras.layers.Dense(1)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, image, training=None):
        x = self.conv_1(image)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = self.conv_blocks(x, training=training)
        x = self.flatten(x)
        x = self.fc_1(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        return self.fc_2(x)
