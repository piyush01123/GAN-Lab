
"""
An implementation of Goodfellow's original GAN paper https://arxiv.org/abs/1406.2661
"""

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model


def get_real_batch(batch_size):
    pass

class GAN:
    def __init__(self):
        pass
    def generator(self):
        pass
    def discriminator(self):
        pass
    def make_gan(self):
        self.G  = self.generator()
        self.D = self.discriminator()
        self.AM = Model(inputs=self.G.input,outputs=self.D.output)
    def train(self):
        writer = tf.summary.FileWriter(self.logdir)
        self.make_gan()
        for iter in range(self.num_iters):
            real = get_real_batch(self.batch_size)
            noise = np.random.normal(0, 1, size=(self.batch_size, self.noise_dim))
            images_fake = self.G.predict(noise)
            real_fake_X = np.stack(real, fake)
            real_fake_Y = np.stack(np.ones(self.batch_size), np.zeros(self.batch_size))
            D_loss = self.D.train_on_batch(real_fake_X, real_fake_Y)
            AM_loss = self.AM.train_on_batch(fake, np.ones(self.batch_size))
            tf.summary.scalar('Disc_loss', D_loss)
            tf.summary.scalar('AM_loss', AM_loss)
            tf.summary.image('Gen_Images', fake)
            if iter%100==0:
                self.save_images(fake)
    def save_images(self, images):
        idx = np.random.choice(self.batch_size, 16)
        chosen_images = images[idx]
        figure = np.zeros(4*28, 4*28)
        for i, image in enumerate(chosen_images):
            r = i//4
            c = i%4
            figure[28*r:28*(r+1),28*c:28*(c+1)] = figure
        Image.fromarray(figure).
