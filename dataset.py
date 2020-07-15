import os
from glob import glob

import numpy as np
import tensorflow as tf


def read_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image.set_shape([None, None, 3])
    return tf.cast(image, dtype=tf.float32)


def prepare_image(crop_size, downscale_factor):
    def _prepare_image(image_path):
        image = read_image(image_path)
        hr_image = tf.image.random_crop(image, [crop_size, crop_size, 3])
        lr_image = tf.image.resize(
            hr_image, 
            [crop_size//downscale_factor, crop_size//downscale_factor], 
            'bicubic')
        hr_image = hr_image / 127.5 - 1
        return lr_image, hr_image
    return _prepare_image


def get_dataset(image_dir, batch_size, crop_size, downscale_factor):
    images = glob(os.path.join(image_dir, '*'))
    np.random.shuffle(images)

    autotune = tf.data.experimental.AUTOTUNE
    dataset = tf.data.Dataset.from_tensor_slices(images)
    dataset = dataset.shuffle(512)
    dataset = dataset.map(prepare_image(crop_size, downscale_factor), num_parallel_calls=autotune)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(autotune)
    return dataset
