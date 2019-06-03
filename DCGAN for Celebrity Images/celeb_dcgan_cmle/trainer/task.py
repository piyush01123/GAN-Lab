import argparse
import logging
import os

from . import model
from . import utils

import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps',
                        type=int,
                        default=40000,
                        required=False,
                        help='Number of steps or epochs to train'
                        )
    parser.add_argument('--save-dir',
                        type=str,
                        required=True,
                        help='Generated image save directory local or GCS'
                        )
    parser.add_argument('--save-interval',
                        type=int,
                        default=100,
                        help='Generated image save interval'
                        )
    parser.add_argument('--batch-size',
                        type=int,
                        default=32,
                        help='Generated image save directory local or GCS'
                        )
    parser.add_argument('--verbosity',
                        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
                        default='ERROR'
                        )
    parser.add_argument('--job-dir',
                        required=False,
                        help='Some problem with submit bash command forces me to write this'
                        )
    return parser.parse_args()


def train_and_evaluate(hparams):
    utils.download_dataset()
    dcgan = model.DCGAN_CELEB(num_steps=hparams.num_steps,
                        save_dir=hparams.save_dir,
                        save_interval=hparams.save_interval,
                        batch_size=hparams.batch_size
                        )
    # dcgan.train()


if __name__ == '__main__':
  args = get_args()
  tf.logging.set_verbosity(args.verbosity)
  hparams = hparam.HParams(**args.__dict__)
  train_and_evaluate(hparams)
