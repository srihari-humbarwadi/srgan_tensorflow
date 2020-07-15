import argparse
import os

import numpy as np
from skimage.io import imsave
import tensorflow as tf

from srgan.model import SRGan


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i',
                        '--input_image',
                        type=str,
                        required=True,
                        metavar='',
                        help='Input image')

    parser.add_argument('-m',
                        '--saved_model_dir',
                        type=str,
                        required=True,
                        metavar='',
                        help='Saved model location')

    args = parser.parse_args()
    return args


def prepare_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image.set_shape([None, None, 3])
    return tf.cast(image[None, ...], dtype=tf.float32)


def main():
    args = _parse_args()

    model = tf.saved_model.load(args.saved_model_dir)
    
    lr_image = prepare_image(args.input_image)
    sr_image = model(lr_image)[0]
    sr_image = np.uint8((sr_image + 1) * 127.5)

    file_name = 'super_resolved_' + os.path.basename(args.input_image)
    imsave(os.path.join('results', file_name), sr_image)
    print('Saved result at {}'.format(os.path.join('results', file_name)))


if __name__ == '__main__':
    main()
