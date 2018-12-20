from keras.models import Sequential
from keras.optimizers import RMSprop
import numpy as np

class GAN:
    """"
    Convenience class for GANs. This class can be used to quickly build GANs with
    suitable Generator and Discriminator models, learning rates and data.
    """
    def __init__(self, generator, discriminator, data, noise_dim, G_lr,  D_lr, AM_lr, batch_size = 50, num_steps = 100):
        """
        GAN constructor using G, D and learning rates.
        Args:
            generator: Keras model object
            discriminator: Keras model object
            data: training data
            noise_dim: Dimension of Noise Vector which is same as input shape of G
            G_lr, D_lr, AM_lr: Learning rates for G, D and AM respectively
            batch_size: Batch Size for training
            num_steps: Number of training steps
        Returns:
            Instantiated GAN object
        """
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
        """
        Makes GAN ie does the mechanics of compiling Generator, Discriminator and Adversarial models
        """
        opt = RMSprop(lr=self.G_lr, clipvalue=1.0, decay=6e-8)
        self.D.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        opt = RMSprop(lr=self.D_lr, clipvalue=1.0, decay=3e-8)
        self.G.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        self.AM = Sequential()
        self.AM.add(self.G)
        self.AM.add(self.D)
        opt = RMSprop(lr=self.AM_lr, decay=3e-8)
        self.AM.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        print(self.G.summary(), self.D.summary(), self.AM.summary())
        print(self.G.input_shape, self.G.output_shape, self.D.input_shape, self.D.output_shape, self.AM.input_shape, self.AM.output_shape)

    def train(self):
        """
        Trains GAN ie trains G and D simultaneously.
        To train a GAN we have G which maps (None, noise_dim) to (None, data.input_shape)
        and D which maps (None, data.input_shape) to (None, 1)
        For example in example.py we have a G mapping (None, 100) to (None, 28, 28, 1)
        and D which maps (None, 28, 28, 1) to (None, 1).
        Essentially at each step we are teaching D to differentiate between fake and real samples
        and then AM (G+D) to create fake samples which are able to fool D.
        """
        self.makeGAN()
        for i in range(self.num_steps):
            noise = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.noise_dim])
            fake = self.G.predict(noise)
            real = self.data[np.random.randint(0, self.data.shape[0], self.batch_size), :, :, :]
            train_x = np.concatenate((real, fake))
            train_y = np.concatenate((np.ones((self.batch_size, 1)), np.zeros((self.batch_size, 1))))
            d_loss = self.D.train_on_batch(train_x, train_y)

            noise = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.noise_dim])
            Y = np.ones((self.batch_size, 1)) # we are forcing the Adversarial model to have output class 1 for fake samples
            a_loss = self.AM.train_on_batch(noise, Y)

            print('D Loss = %f, Accuracy of Discriminator or accuracy with which D can tell fake from real = %f' %(d_loss[0], d_loss[1]))
            print('Adversarial Loss = %f, Accuracy of Adversarial model or accuracy with which G can fake D = %f' %(a_loss[0], a_loss[1]))
