
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model

class CycleGAN:
    def __init__(self):
        pass

    def create_generator(self):
        pass

    def create_discriminator(self):
        pass

    def create_gan(self):
        self.G_AB = self.create_generator()
        self.G_BA = self.create_generator()
        self.D_B = self.create_discriminator()
        self.D_A = self.create_discriminator()

        img_a = Input(shape=self.input_shape)
        img_b = self.G_AB(img_a)
        self.D_B.trainable = False
        validity = self.D_B(img_b)
        self.GAN_AB = Model(inputs=z, outputs=validity)
        self.GAN_AB.summary()
        self.GAN_AB.compile()
        self.D_B.trainable = True

        img_b = Inupt(shape=(self.input_shape))
        img_a = self.G_BA(img_b)
        self.D_A.trainable = False
        validity = self.D_A(img_a)
        self.GAN_BA = Model(inputs=z, outputs=validity)
        self.GAN_AB.compile()
        self.D_A.trainable = True

        img_a = Input(shape=self.input_shape)
        img_b = self.G_AB(img_a)
        img_a_hat = self.G_BA(img_b)
        self.GAN_ABA = Model(inputs=img_a, outputs=img_a_hat)
        self.GAN_ABA.summary()
        self.GAN_ABA.compile(loss='mse')

        img_b = Input(shape=self.input_shape)
        img_a = self.G_BA(img_b)
        img_b_hat = self.G_AB(img_a)
        self.GAN_BAB = Model(inputs=img_b, outputs=img_b_hat)
        self.GAN_BAB.summary()
        self.GAN_BAB.compile(loss='mse')

    def train(self):
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))
        for step in range(num_steps):
            real_a = self.get_real_images_batch(self.dataset_A)
            real_b = self.get_real_images_batch(self.dataset_B)
            fake_b = self.G_AB.predict(real_a)
            fake_a = self.G_BA.predict(real_b)
            fake_a_hat = self.G_ABA.predict(real_a)
            fake_b_hat = self.G_BAB.predict(real_b)

            loss_D_B_real = self.D_B.train_on_batch(real_b, valid)
            loss_D_B_fake = self.D_B.train_on_batch(fake_b, fake)
            loss_D_B = 0.5*np.add(loss_D_B_real, loss_D_B_fake)

            loss_D_A_real = self.D_A.train_on_batch(real_a, valid)
            loss_D_A_fake = self.D_A.train_on_batch(fake_a, fake)
            loss_D_A = 0.5*np.add(loss_D_A_real, loss_D_A_fake)

            self.GAN_AB.train_on_batch(fake_b, valid)
            self.GAN_BA.train_on_batch(fake_a, valid)

            self.GAN_ABA.train_on_batch(real_a, fake_a_hat)
            self.GAN_BAB.train_on_batch(real_b, fake_b_hat)

    def save_images(self):
        pass

    def create_batch(dataset):
        pass

if __name__=='__main__':
    cyclegan = CycleGAN()
    cyclegan.train()
