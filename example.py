from gan  import GAN
import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.datasets import mnist

class exampleGAN(GAN):
    def __init__(self):
        """
        Instantiates an example GAN. Architecture is same as dcgan.py
        """
        self.input_shape, self.noise_dim, self.G_lr,  self.D_lr, self.AM_lr  = (28, 28, 1), 100, .004, .008, .001
        self.dropout, self.generated_dim = 0.4, 7
        self.data = np.expand_dims(mnist.load_data()[0][0].astype('float32')/255, axis = -1)
        print('Data Shape: ', self.data.shape)
        self.G, self.D = self.Genr(), self.Discr()
        super().__init__(self.G, self.D, self.data, self.noise_dim, self.G_lr,  self.D_lr, self.AM_lr)

    def Discr(self):
        """
        Discriminator for exampleGAN
        """
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

    def Genr(self):
        """
        Generator for exampleGAN
        """
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

def main():
    """
    Instantiate and train an exampleGAN
    """
    mygan = exampleGAN()
    mygan.makeGAN()
    mygan.train()

if __name__ == '__main__':
    main()
