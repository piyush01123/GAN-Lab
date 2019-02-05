
from tensorflow.keras.models  import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import numpy as np
from PIL import Image

class DCGAN:
    def __init__(self, latent_dim, input_shape):
        self.latent_dim = latent_dim
        self.input_shape = input_shape

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
        model.add(tf.keras.layers.Dense(1)) #notice there's no sigmoid
        model.summary()
        return model

class MNISTGAN:
    def __init__(self, save_interval = 10):
        (self.data, _), (_, _) = mnist.load_data()
        self.image_size = self.data.shape[1:]
        self.input_shape = self.image_size+(1,)
        self.latent_dim = 100
        self.dcgan = DCGAN(latent_dim=self.latent_dim, input_shape=self.input_shape)
        self.discriminator = self.dcgan.make_discriminator_model()
        self.generator = self.dcgan.make_generator_model()
        self.save_interval = save_interval

    def train(self):
        pass

if __name__=='__main__':
    mnistgan = MNISTGAN()
    mnistgan.train()
