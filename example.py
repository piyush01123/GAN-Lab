from gan  import GAN
import numpy as np

class exampleGAN(GAN):
    def __init__(self):
        self.G, self.D, , self.noise_dim, self.G_lr,  self.D_lr, self.AM_lr =
        self.generator(), self.discriminator(), 100, .004, .008, .001
        self.input_shape = 28, 28, 1
        self.data = np.expand_dims(mnist.load_data()[0][0].astype('float32')/255, axis = -1)
        super().__init__(self.G, self.D, self.data, self.noise_dim, self.G_lr,  self.D_lr, self.AM_lr)

    def generator(self):
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

    def discriminator(self):
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
    myGAN = exampleGAN()
    myGAN.train()

if __name__ == '__main__':
    main()
