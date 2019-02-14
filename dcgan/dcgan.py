
from tensorflow.keras.models  import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import imageio
import glob

class DCGAN:
    def __init__(self):
        (self.data, _), (_, _) = mnist.load_data()
        self.latent_dim = 100
        self.input_shape = self.data.shape[1:] + (1,)
        self.batch_size = 32
        self.num_steps = 4000
        self.save_dir = 'plots/'
        self.save_interval=100
        self.preprocess_data()
        self.make_gan()

    def make_generator_model(self):
        model = Sequential()
        model.add(Dense(7*7*256, use_bias=False, input_shape=(self.latent_dim,)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((7, 7, 256)))
        model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        model.summary()
        model = Model(model.inputs, model.outputs)
        return model

    def make_discriminator_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.input_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))


        model.summary()
        model = Model(model.inputs, model.outputs)
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
            )

        self.D.trainable = True # To avoid all the dirty warnings


    def train(self):
        valids = np.ones((self.batch_size, 1))
        fakes = np.zeros((self.batch_size, 1))
        for step in range(self.num_steps):
            images_real = self.get_real_images_batch()
            d_loss_real = self.D.train_on_batch(images_real, valids)
            noise = np.random.normal(0, 1, size=[self.batch_size, self.latent_dim])
            images_fake = self.G.predict(noise)
            d_loss_fake = self.D.train_on_batch(images_fake, fakes)
            d_loss, d_acc = 0.5*np.add(d_loss_real, d_loss_fake)

            g_loss = self.AM.train_on_batch(noise, valids)

            loss_msg = {'Step': step, 'D_loss': d_loss, 'D_acc': 100*d_acc, 'G_loss': g_loss}
            print(loss_msg)

            if step%self.save_interval==0:
                self.plot_images(step)

        self.plot_images(step)
        self.D.save('discriminator.h5')
        self.G.save('generator.h5')
        self.AM.save('adversarial.h5')
        self.create_progression_gif()

    def plot_images(self, step, num_images = 16):
        noise = np.random.normal(0, 1, size=[num_images, self.latent_dim])
        fake_images = self.G.predict(noise)
        fake_images = fake_images*0.5 + 0.5 # rescale to (0, 1)
        fig, axes = plt.subplots(4, 4)
        for i, img in enumerate(fake_images):
            r, c = i//4, i%4
            axes[r, c].imshow(img[:,:,0], cmap='gray')
            axes[r, c].axis('off')
        fig.savefig(self.save_dir+'step_%s.png' %step)

    def create_progression_gif(self):
        with imageio.get_writer(self.save_dir+'progression.gif', mode='I') as writer:
            filenames = glob.glob(self.save_dir+'step_*.png')
            filenames = sorted(filenames, key = lambda k: int(k[11:-4]))
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

    def preprocess_data(self):
        # Scale in -1 to 1
        self.data = self.data.astype('float32')/127.5-1
        self.data = np.expand_dims(self.data, axis = -1)

    def get_real_images_batch(self):
        idx = np.random.randint(0, self.data.shape[0], self.batch_size)
        return self.data[idx]


if __name__=='__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    dcgan = DCGAN()
    dcgan.train()
