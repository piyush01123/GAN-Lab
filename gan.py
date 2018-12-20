from keras.models import Sequential
from keras.optimizers import RMSprop
import numpy as np

class GAN:
    def __init__(self, generator, discriminator, data, noise_dim, G_lr,  D_lr, AM_lr, batch_size = 50, num_steps = 100):
        self.G = generator
        self.D = discriminator
        self.data = data
        self.noise_dim = noise_dim
        self.G_lr = G_lr
        self.D_lr = D_lr
        self.AM_lr = AM_lr
        self.batch_size = batch_size
        self.num_steps = num_steps

    def makeGAN(self):
        opt = RMSprop(lr=self.G_lr, clipvalue=1.0, decay=6e-8)
        self.D.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        opt = RMSprop(lr=self.D_lr, clipvalue=1.0, decay=3e-8)
        self.G.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        print(self.G.summary(), self.D.summary())
        self.AM = Sequential()
        self.AM.add(self.G)
        self.AM.add(self.D)
        print(self.AM.summary())
        opt = RMSprop(lr=self.AM_lr, decay=3e-8)
        self.AM.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    def train(self):
        self.makeGAN()
        for i in range(self.num_steps):
            noise = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.noise_dim])
            fake = self.G.predict(noise)
            real = self.data[np.random.randint(0, self.data.shape[0], self.batch_size), :, :, :]
            train_x = np.concatenate((real, fake))
            train_y = np.concatenate((np.ones((self.batch_size, 1)), np.zeros((self.batch_size, 1))))
            d_loss = self.D.train_on_batch(train_x, train_y)

            noise = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.noise_dim])
            Y = np.ones((self.batch_size, 1))
            a_loss = self.AM.train_on_batch(noise, Y)

            print('D Loss = %f, Accuracy of Discriminator or accuracy with which D can say fake from real = %f' %(d_loss[0], d_loss[1]))
            print('Adversarial Loss = %f, Accuracy of Adversarial model or accuracy with which G can fake D = %f' %(a_loss[0], a_loss[1]))
