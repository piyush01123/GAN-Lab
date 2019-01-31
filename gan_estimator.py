
from tensorflow.contrib import gan as tfgan
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
from PIL import Image

class DCGAN:
    def __init__(self):
        self.input_shape = 28, 28, 1
        self.dropout = 0.4
        self.noise_dim = 100
        self.generated_dim = 7
        self.batch_size = 50
        self.num_steps = 100
        self.model_dir = 'checkpoints/'

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train.astype('float32')/255, x_test.astype('float32')/255
        X = np.concatenate((x_train, x_test), axis = 0)
        X = np.expand_dims(X, axis = -1)
        self.data = X

    def Discriminator(self):
        D = Sequential()
        D.add(Conv2D(filters = 64, kernel_size = 5, strides = 2, input_shape = self.input_shape,
        padding = 'same', activation = LeakyReLU(alpha=0.3)))
        D.add(Dropout(self.dropout))
        D.add(Conv2D(filters = 128, kernel_size = 5, strides = 2, input_shape = self.input_shape,
        padding = 'same', activation = LeakyReLU(alpha=0.2)))
        D.add(Dropout(self.dropout))
        D.add(Conv2D(filters = 256, kernel_size = 5, strides = 2, input_shape = self.input_shape,
        padding = 'same', activation = LeakyReLU(alpha=0.2)))
        D.add(Dropout(self.dropout))
        D.add(Conv2D(filters = 512, kernel_size = 5, strides = 1, input_shape = self.input_shape,
        padding = 'same', activation = LeakyReLU(alpha=0.2)))
        D.add(Dropout(self.dropout))
        D.add(Flatten())
        D.add(Dense(1))
        D.add(Activation('sigmoid'))
        return D

    def Generator(self):
        G = Sequential()
        G.add(Dense(self.generated_dim*self.generated_dim*256, input_dim = self.noise_dim))
        G.add(BatchNormalization(momentum = 0.9))
        G.add(Activation('relu'))
        G.add(Reshape((self.generated_dim, self.generated_dim, 256)))
        G.add(Dropout(self.dropout))
        G.add(UpSampling2D())
        G.add(Conv2DTranspose(filters = 128, kernel_size = 5, padding = 'same'))
        G.add(BatchNormalization(momentum = 0.9))
        G.add(Activation('relu'))
        G.add(UpSampling2D())
        G.add(Conv2DTranspose(filters = 64, kernel_size = 5, padding = 'same'))
        G.add(BatchNormalization(momentum = 0.9))
        G.add(Activation('relu'))
        G.add(Conv2DTranspose(filters = 32, kernel_size = 5, padding = 'same'))
        G.add(BatchNormalization(momentum = 0.9))
        G.add(Activation('relu'))
        G.add(Conv2DTranspose(filters = 1, kernel_size = 5, padding = 'same'))
        G.add(Activation('sigmoid'))
        return G

    def train_input_function(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.data)
        dataset = dataset.shuffle(1000).repeat().batch(self.batch_size)
        images = dataset.make_one_shot_iterator().get_next()
        tf_session = K.get_session()
        with tf_session:
            noise = tf.random_normal([self.batch_size, self.noise_dim])
        return noise, images

    def predict_input_function(self):
        noise = tf.random_normal([self.batch_size, self.noise_dim])
        return noise

    def make_estimator(self):
        print(self.Generator().summary())
        print(self.Discriminator().summary())
        return tfgan.estimator.GANEstimator(
        self.model_dir,
        generator_fn=self.Generator(),
        discriminator_fn=self.Discriminator(),
        generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
        discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
        generator_optimizer=tf.train.AdamOptimizer(0.1, 0.5)
        )

    def train(self):
        self.gan_estimator = self.make_estimator()
        print('GAN estimator', self.gan_estimator)
        self.gan_estimator.train(self.train_input_function, max_steps=self.num_steps)

    def plot_images(self, i, num_images = 16):
        pass


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train()
