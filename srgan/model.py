import tensorflow as tf
from srgan.networks import Generator, Discriminator
from losses import VGGLoss
from metrics import PSNR, SSIM


class SRGan(tf.keras.Model):

    def __init__(self, upscale_factor=4, generator_weights=None, **kwargs):
        super(SRGan, self).__init__(**kwargs)
        self.generator = Generator(weights=generator_weights)
        self.discriminator = Discriminator()
        self.vgg_loss = VGGLoss()

    def compile(self, d_optimizer, g_optimizer, loss_fn, **kwargs):
        super(SRGan, self).compile(**kwargs)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self._psnr = PSNR(max_val=1.0)
        self._ssim = SSIM(max_val=1.0)

    def train_step(self, data):
        lr_images, hr_images = data
        batch_size = tf.shape(lr_images)[0]

        ones = tf.ones([batch_size])
        zeros = tf.zeros([batch_size])

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            sr_images = self.generator(lr_images, training=True)

            fake_logits = self.discriminator(sr_images, training=True)
            real_logits = self.discriminator(hr_images, training=True)

            d_loss_fake = tf.reduce_mean(self.loss_fn(zeros, fake_logits))
            d_loss_real = tf.reduce_mean(self.loss_fn(ones, real_logits))
            d_loss = d_loss_fake + d_loss_real

            content_loss = self.vgg_loss(hr_images, sr_images)
            g_loss = tf.reduce_mean(self.loss_fn(ones, fake_logits))
            perceptual_loss = content_loss + 1e-3 * g_loss

            d_loss_scaled = \
                d_loss / self.distribute_strategy.num_replicas_in_sync
            perceptual_loss_scaled = \
                perceptual_loss / self.distribute_strategy.num_replicas_in_sync

        d_grads = d_tape.gradient(d_loss_scaled,
                                  self.discriminator.trainable_weights)
        g_grads = g_tape.gradient(perceptual_loss_scaled,
                                  self.generator.trainable_weights)

        self.d_optimizer.apply_gradients(
            zip(d_grads, self.discriminator.trainable_weights))
        self.g_optimizer.apply_gradients(
            zip(g_grads, self.generator.trainable_weights))

        self._psnr.update_state(hr_images, sr_images)
        self._ssim.update_state(hr_images, sr_images)

        return {
            'psnr': self._psnr.result(),
            'ssim': self._ssim.result(),
            'perceptual_loss': perceptual_loss,
            'content_loss': content_loss,
            'g_loss': g_loss,
            'd_loss_real': d_loss_real,
            'd_loss_fake': d_loss_fake
        }
