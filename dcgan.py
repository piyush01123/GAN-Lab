
from tensorflow.keras.models  import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import numpy as np
from PIL import Image

class DCGAN:
    def __init__(self):
        (self.data, _), (_, _) = mnist.load_data()
        self.latent_dim = 100
        self.input_shape = None, None, 3

    def make_generator_model(self):
        model = Sequential()
        model.add(Dense(7*7*256, use_bias=False, input_shape=(self.latent_dim,)))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Reshape((7, 7, 256)))
        model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        model.summary()
        return model

    def make_discriminator_model(self):
        model = Sequential()
        model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                         input_shape=self.input_shape))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1))
        model.summary()
        return model

    def make_adversarial_model(self):
        D = self.make_discriminator_model()
        G = self.make_generator_model()
        AM = Sequential()
        AM.add(G)
        AM.add(D)
        AM.summary()
        return D, G, AM

    def make_gan(self):
        self.make_discriminator_model()
        exit('EXITING')
        self.D, self.AM, self.G = self.make_adversarial_model()

    def train(self):
        self.make_gan()

if __name__=='__main__':
    dcgan = DCGAN()
    dcgan.train()
