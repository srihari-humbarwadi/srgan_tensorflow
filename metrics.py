import tensorflow as tf


class PSNR(tf.metrics.Mean):

    def __init__(self, max_val=1.0, rescale_images=True, **kwargs):
        super(PSNR, self).__init__(name='psnr', **kwargs)
        self.rescale_images = rescale_images
        self.max_val = max_val

    def update_state(self, y_true, y_pred, *args, **kwargs):
        if self.rescale_images:
            y_true = (y_true + 1) * self.max_val / 2
            y_pred = (y_pred + 1) * self.max_val / 2
        psnr = tf.image.psnr(y_true, y_pred, self.max_val)
        super(PSNR, self).update_state(psnr, *args, **kwargs)


class SSIM(tf.metrics.Mean):

    def __init__(self, max_val=1.0, rescale_images=True, **kwargs):
        super(SSIM, self).__init__(name='ssim', **kwargs)
        self.rescale_images = rescale_images
        self.max_val = max_val

    def update_state(self, y_true, y_pred, *args, **kwargs):
        if self.rescale_images:
            y_true = (y_true + 1) * self.max_val / 2
            y_pred = (y_pred + 1) * self.max_val / 2
        ssim = tf.image.ssim(y_true, y_pred, self.max_val)
        super(SSIM, self).update_state(ssim, *args, **kwargs)
