
from tensorflow.keras.models  import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import numpy as np
from PIL import Image

class DCGAN:
    def __init__(self):
        (self.data, _), (_, _) = mnist.load_data()
        self.latent_dim = 100
        self.input_shape = self.data.shape[1:] + (1,)
        self.num_steps = 4000
        self.manage_data()
        self.make_gan()

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
        model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                         input_shape=self.input_shape))
        model.add(LeakyReLU())
        model.add(Dropout(0.3))
        model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(LeakyReLU())
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(1))
        model.summary()
        return model

    def make_gan(self):
        optimizer = Adam(0.0002, 0.5)
        self.D = self.make_discriminator_model()
        self.D.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
            )

        self.G = self.make_generator_model()
        z = Input(shape=(self.latent_dim, ))
        img = self.G(z)
        self.D.trainable = False # This is important, mention it in blog
        validity = self.D(img)
        self.AM = Model(z, validity)
        self.AM.summary()
        self.AM.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
            )

    def train(self):
        valids = np.ones((self.batch_size, 1))
        fakes = np.zeros((self.batch_size, 1))
        for step in range(self.num_steps):
            images_real = self.get_real_images_batch()
            d_loss_real = self.train_on_batch(images_real, valids)
            noise = np.random.normal(0, 1, size=[self.batch_size, self.noise_dim])
            images_fake = self.G.predict(noise)
            d_loss_fake = self.train_on_batch(images_fake, fakes)
            d_loss, d_acc = 0.5*np.add(d_loss_real, d_loss_fake)

            g_loss = self.AM.train_on_batch(noise, valids)

            loss_msg = {'Step': step, 'D_loss': d_loss, 'D_acc': d_acc, 'G_loss', g_loss}
            print(loss_msg)

            if step%10==0:
                self.plot_images(step)

    def plot_images(self, step, num_images = 16, save_dir = 'plots/'):
        noise = np.random.normal(0, 1, size=[num_images, self.noise_dim])
        fake_images = self.G.predict(noise)
        fake_images = np.squeeze(fake_images)
        for j, img in enumerate(fake_images):
            image = Image.fromarray(img.astype('uint8'))
            filename = save_dir+"step%d_image%d.png" %(i, j)
            image.save(filename)

    def manage_data(self):
        # Scale in -1 to 1
        self.data = self.data.astype('float32')/127.5-1
        self.data = np.expand_dims(self.data, axis = -1)

    def get_real_images_batch(self):
        idx = np.random.randint(0, self.data.shape[0], self.batch_size)
        return self.data[idx]



if __name__=='__main__':
    dcgan = DCGAN()
    dcgan.train()
