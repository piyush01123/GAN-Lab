
fromt tensorflow.keras.layers import *
fromt tensorflow.keras.models import *

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
        self.GAN_AB.compile()
        self.D_B.trainable = True

        img_b = Inupt(shape=(self.latent_dim,))
        img_a = self.G_BA(img_b)
        self.D_A.trainable = False
        validity = self.D_A(img_a)
        self.GAN_BA = Model(inputs=z, outputs=validity)
        self.GAN_AB.compile()
        self.D_A.trainable = True

        img_a = Input(shape=self.input_shape)
        img_b = self.G_AB(img_a)
        img_a_hat = self.G_BA(img_b)
        loss_tensor = diff_fn(img_a, img_a_hat) ## TODO: implement diff_fn
        self.GAN_AA = Model(inputs=img_a, outputs=img_a_hat)
        self.GAN_AA.compile(loss=loss_tensor, )

        img_b = Input(shape=self.input_shape)
        img_a = self.G_BA(img_b)
        img_b_hat = self.G_AB(img_a)
        loss_tensor = diff_fn(img_b, img_b_hat) ## TODO: implement diff_fn
        self.GAN_BB = Model(inputs=img_b, outputs=img_b_hat)
        self.GAN_AA.compile(loss=loss_tensor, )

    def train(self):
        for step in range(num_steps):


    def save_images(self):
        pass

    def create_batch(dataset):
        pass

if __name__=='__main__':
    cyclegan = CycleGAN()
    cyclegan.train()
