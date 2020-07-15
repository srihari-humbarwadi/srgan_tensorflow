import argparse
import os

import tensorflow as tf

from dataset import get_dataset
from metrics import PSNR, SSIM
from srgan.networks import Generator


os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--iterations',
                        type=int,
                        required=True,
                        metavar='',
                        help='Total training iterations')

    parser.add_argument('-s',
                        '--step_per_epoch',
                        type=int,
                        required=True,
                        metavar='',
                        help='Iterations per epoch')

    parser.add_argument('-b',
                        '--batch_size',
                        type=int,
                        required=True,
                        metavar='',
                        help='Batch size')

    parser.add_argument('-c',
                        '--crop_size',
                        type=int,
                        required=True,
                        metavar='',
                        help='Crop size')

    parser.add_argument('-f',
                        '--downscale_factor',
                        type=int,
                        required=True,
                        metavar='',
                        help='Downscale factor')

    parser.add_argument('-d',
                        '--image_dir',
                        type=str,
                        required=True,
                        metavar='',
                        help='Directory containing images')

    parser.add_argument('-m',
                        '--model_dir',
                        type=str,
                        required=True,
                        metavar='',
                        help='Model directory')
    
    parser.add_argument('-w',
                        '--weights_path',
                        type=str,
                        required=False,
                        metavar='',
                        help='Load weights from')
    
    args = parser.parse_args()
    return args


def main():
    args = _parse_args()

    model_dir = os.path.join(args.model_dir, 'srresnet')

    train_dataset = get_dataset(
        args.image_dir,
        args.batch_size,
        args.crop_size,
        args.downscale_factor)

    mse_loss = tf.losses.MeanSquaredError()
    optimizer = tf.optimizers.Adam(1e-4)
    model = Generator(upscale_factor=args.downscale_factor)
    model.build((1, None, None, 3))
    
    model.compile(optimizer=optimizer, loss=mse_loss, metrics=[
                  PSNR(max_val=1.0), SSIM(max_val=1.0)])
    
    if args.weights_path:
        model.load_weights(args.weights_path)
        print('Loaded weights from {}'.format(args.weights_path))

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(model_dir, 'logs'),
                                       update_freq=500),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir,
                         'srresnet_weights_epoch_') + '{epoch}',
            save_weights_only=True,
            save_best_only=False,
            monitor='loss',
            verbose=1,
            save_freq=args.step_per_epoch // 2),
    ]

    epochs = args.iterations // args.step_per_epoch
    
    model.fit(train_dataset,
              epochs=epochs,
              steps_per_epoch=args.step_per_epoch,
              callbacks=callbacks)


if __name__ == '__main__':
    strategy = tf.distribute.OneDeviceStrategy('/gpu:0')

    with strategy.scope():
        main()
