import tensorflow as tf


class VGGLoss(tf.losses.Loss):

    def __init__(self, rescale_images=True, **kwargs):
        super(VGGLoss, self).__init__(reduction='none', **kwargs)
        self.rescale_images = rescale_images
        self._scale = tf.convert_to_tensor(12.75)
        self._build_vgg_network()
        self._mse = tf.losses.MeanSquaredError(reduction='none')

    def _build_vgg_network(self):
        vgg_model = tf.keras.applications.VGG19(include_top=False)
        vgg_model.trainable = False
        vgg_model.compile()
        self._compute_vgg_features = tf.keras.Model(
            vgg_model.input,
            vgg_model.get_layer('block5_conv4').output)

    def call(self, y_true, y_pred):
        if self.rescale_images:
            y_true = (y_true + 1) * 127.5
            y_pred = (y_pred + 1) * 127.5

        y_true = tf.keras.applications.vgg19.preprocess_input(y_true)
        y_pred = tf.keras.applications.vgg19.preprocess_input(y_pred)

        y_true_features = self._compute_vgg_features(y_true) / self._scale
        y_pred_features = self._compute_vgg_features(y_pred) / self._scale

        loss = self._mse(y_true_features, y_pred_features)
        return tf.reduce_mean(loss)
