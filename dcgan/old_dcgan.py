from keras.models import Sequential
from keras.layers import *
from keras.optimizers import RMSprop
from keras.datasets import mnist
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

    def makeGAN(self):
        self.D = self.Discriminator()
        print(self.D.summary())
        opt = RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)
        self.D.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        self.G = self.Generator()
        print(self.G.summary())
        opt = RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8)
        self.G.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        self.AM = Sequential()
        self.AM.add(self.G)
        self.AM.add(self.D)
        print(self.AM.summary())
        opt = RMSprop(lr=0.0001, decay=3e-8)
        self.AM.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    def train(self):
        self.makeGAN()
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train.astype('float32')/255, x_test.astype('float32')/255
        X = np.concatenate((x_train, x_test), axis = 0)
        X = np.expand_dims(X, axis = -1)

        for i in range(self.num_steps):
            noise = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.noise_dim])
            images_fake = self.G.predict(noise)
            images_real = X[np.random.randint(0, X.shape[0], self.batch_size), :, :, :]
            train_x = np.concatenate((images_real, images_fake))
            train_y = np.concatenate((np.ones((self.batch_size, 1)), np.zeros((self.batch_size, 1))))
            d_loss = self.D.train_on_batch(train_x, train_y)

            noise = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.noise_dim])
            Y = np.ones((self.batch_size, 1))
            a_loss = self.AM.train_on_batch(noise, Y)

            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)

            if i%10==0:
                self.plot_images(i)

    def plot_images(self, i, num_images = 16):
        noise = np.random.uniform(-1.0, 1.0, size=[num_images, self.noise_dim])
        fake_images = self.G.predict(noise)
        fake_images = np.squeeze(fake_images)
        for j, img in enumerate(fake_images):
            image = Image.fromarray(img.astype('uint8'))
            filename = "plots/iteration_%d_image%d.png" %(i, j)
            image.save(filename)


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train()
