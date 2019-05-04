
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from PIL import Image
import tensorflow as tf
import numpy as np
import os, glob


class CycleGAN:
    def __init__(self):
        a_globs = glob.glob('datasets/vangogh2photo/*A/*.jpg')
        b_globs = glob.glob('datasets/vangogh2photo/*B/*.jpg')
        self.img_files = dict()
        self.img_files['A'] = np.array(a_globs, dtype=np.str)
        self.img_files['B'] = np.array(b_globs, dtype=np.str)
        self.lambda_cyc = 10
        self.learning_rate = 0.0002
        # self.image_width = 256
        # self.image_height = 256
        self.input_shape = (256, 256, 3)
        self.batch_size = 50
        self.num_steps = 4000
        self.save_dir = 'generated_hd/'
        self.save_interval = 100
        self.logdir = 'logs_hd/'
        self.summary_writer = tf.summary.FileWriter(self.logdir)
        self.checkpoint_dir = 'checkpoints_hd/'
        self.make_gan()


    def create_generator(self):
        # U-Net like architecture.
        return Generator()


    def create_discriminator(self):
        # Down-facing triangle like architecture.
        return Discriminator()


    def make_gan(self):
        self.G_AB = self.create_generator() #A->B
        self.G_BA = self.create_generator() #B->A
        self.D_B = self.create_discriminator()
        self.D_A = self.create_discriminator()

        img_a = Input(shape=self.input_shape)
        img_b = self.G_AB(img_a)
        self.D_B.trainable = False
        validity_b = self.D_B(img_b)
        self.GAN_AB = Model(inputs=z, outputs=validity_b)
        self.GAN_AB.summary()
        self.GAN_AB.compile()
        self.D_B.trainable = True

        img_b = Inupt(shape=(self.input_shape))
        img_a = self.G_BA(img_b)
        self.D_A.trainable = False
        validity_a = self.D_A(img_a)
        self.GAN_BA = Model(inputs=z, outputs=validity_a)
        self.GAN_BA.compile()
        self.D_A.trainable = True

        img_a = Input(shape=self.input_shape)
        img_b = Input(shape=self.input_shape)
        img_a_hat = self.G_BA(self.G_AB(img_b))
        img_b_hat = self.G_AB(self.G_BA(img_b))
        self.GAN_cyc = Model(inputs=[img_a, img_b],
                             outputs=[validity_b, validity_a, img_a_hat, img_b_hat],
                            )
        mse_loss = tf.reduce_mean(tf.pow(img_a-img_b, 2))
        self.GAN_cyc.compile(loss=["mse", "mse", "mae", "mae"],
                             loss_weights=[1, 1, self.config.lambda_cyc, self.config.lambda_cyc],
                             learning_rate=self.config.learning_rate
                            )
        # we have two adversarial models and a cyclical consistency enforcer model


    def train(self):
        valids = np.ones((self.batch_size, 1))
        fakes = np.zeros((self.batch_size, 1))
        for step in range(num_steps):
            real_a = self.create_batch(set='A')
            real_b = self.get_real_images_batch(set='B')

            fake_b = self.G_AB.predict(real_a)
            fake_a = self.G_BA.predict(real_b)
            a_hat = self.G_BA.predict(fake_b)
            b_hat = self.G_AB.predict(fake_a)

            loss_D_B_real = self.D_B.train_on_batch(real_b, valids)
            loss_D_B_fake = self.D_B.train_on_batch(fake_b, fakes)
            loss_D_B = 0.5*np.add(loss_D_B_real, loss_D_B_fake)

            loss_D_A_real = self.D_A.train_on_batch(real_a, valids)
            loss_D_A_fake = self.D_A.train_on_batch(fake_a, fakes)
            loss_D_A = 0.5*np.add(loss_D_A_real, loss_D_A_fake)

            gan_ab_loss = self.GAN_AB.train_on_batch(fake_b, valids)
            gan_ba_loss = self.GAN_BA.train_on_batch(fake_a, valids)

            gan_cyc_loss = self.GAN_cyc.train_on_batch(inputs=[real_a, real_b], outputs=[valids, valids, real_a, real_b])

            print("step=%d, cyc_loss=%f, gan_ab_loss=%f, gan_ba_loss=%f, da_loss=%f, db_loss=%f"
                   %(step, gan_cyc_loss, gan_ab_loss, gan_ba_loss, loss_D_A, loss_D_B))

    def save_images(self):
        pass

    def create_batch(self, set):
        idx = np.random.randint(0, self.num_images, self.batch_size)
        sel_img_files =  self.img_files[set][idx]
        data =  np.array([np.array(Image.open(file).resize(self.image_size, Image.BILINEAR))
                for file in sel_img_files
                ])
        data = data.astype('float32')/127.5-1
        return data

if __name__=='__main__':
    cyclegan = CycleGAN()
    cyclegan.train()
