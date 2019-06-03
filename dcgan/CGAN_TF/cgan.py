

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
import tensorflow_datasets as tfds
import os
import matplotlib.pyplot as plt

try:
    assert (tf.__version__).startswith("2.")
except:
    exit("You need 2.0.0 or higher for this program.")

class Generator(tf.keras.models.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.dense1 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='glorot_uniform')
        self.dense2 = tf.keras.layers.Dense(784, activation='tanh', kernel_initializer='glorot_uniform')

    def call(self, z, y):
        # z~(None, 100), y~(None, 10)
        final = self.concat(inputs=[z,  y])
        final = self.dense1(final)
        final = self.dense2(final)
        return final

class Discriminator(tf.keras.models.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.dense1 = tf.keras.layers.Dense(128, kernel_initializer='glorot_uniform')
        self.lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform')

    def call(self, x, y):
        # x~(None, 784), y~(None, 10)
        final = self.concat(inputs=[x,  y])
        final = self.dense1(final)
        final = self.lrelu(final)
        final = self.dense2(final)
        return final

def preprocess(example):
    example['image'] = tf.reshape(example['image'], (-1, 784))
    example['image'] = tf.cast(example['image'], tf.float32)
    example['image'] = (example['image']-127.5)/127.5
    example['label'] = tf.one_hot(example['label'], 10)
    example['image'] = tf.squeeze(example['image'])
    example['label'] = tf.squeeze(example['label'])
    return example

class CGAN:
    def __init__(self):
        self.batch_size = 50
        self.noise_dim=100
        self.dataset = tfds.load('mnist')['train'].map(preprocess).batch(self.batch_size)
        self.G = Generator()
        self.D = Discriminator()
        self.G_opt = tf.keras.optimizers.Adam(lr=1e-4)
        self.D_opt = tf.keras.optimizers.Adam(lr=1e-4)
        self.checkpoint = tf.train.Checkpoint(
        generator=self.G,
        discriminator=self.D,
        gen_optimizer=self.G_opt,
        disc_optimizer=self.D_opt
        )
        self.checkpoint_path = os.path.join("./training_checkpoints", "ckpt")
        self.save_dir = "generated_images/"
        self.num_epochs = 50


    @tf.function
    def train_step(self, img_batch, labels_batch):
        noise = tf.random.normal((self.batch_size, self.noise_dim))
        labels = tf.random.uniform((self.batch_size,), maxval=10, dtype=tf.int32)
        labels = tf.gather(tf.eye(10), labels)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_imgs = self.G(noise, labels)
            probs_fake = self.D(fake_imgs, labels)
            d_loss_fake = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(probs_fake), probs_fake)

            probs_real = self.D(img_batch, labels_batch)
            d_loss_real = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(probs_real), probs_real)

            d_loss = 0.5 * (d_loss_real + d_loss_fake)

            g_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(probs_fake), probs_fake)

        gen_grads = gen_tape.gradient(g_loss, self.G.trainable_variables)
        disc_grads = disc_tape.gradient(d_loss, self.D.trainable_variables)

        self.G_opt.apply_gradients(zip(gen_grads, self.G.trainable_variables))
        self.D_opt.apply_gradients(zip(disc_grads, self.D.trainable_variables))
        return d_loss, g_loss


    def plot_images(self, name):
        noise = tf.random.normal([16, self.noise_dim])
        labels_ = tf.random.uniform((16,), maxval=10, dtype=tf.int32)
        labels = tf.gather(tf.eye(10), labels_)
        fake_imgs = self.G(noise, labels)
        fig, axes = plt.subplots(4, 4)
        for i, img in enumerate(fake_imgs):
            r, c = i//4, i%4
            axes[r, c].imshow(tf.reshape(img, (-1, 28, 28))[0], cmap='gray')
            axes[r, c].axis('off')
            axes[r, c].title.set_text('Digit: %s' %labels_.numpy()[i])
        fig.savefig(self.save_dir+'%s.png' %name)

    def train(self):
        for epoch in range(self.num_epochs):
            for i, batch in enumerate(self.dataset):
                images_batch = batch['image']
                labels_batch = batch['label']
                d_loss, g_loss = self.train_step(images_batch, labels_batch)
                if (i+1)%50==0:
                    self.plot_images("epoch_%s_batch_%s" %(epoch, i))
                    self.checkpoint.save(self.checkpoint_path)
                    template = 'Epoch {}, , Batch {}, Generator loss {}, Discriminator Loss {}'
                    print (template.format(epoch, i, g_loss, d_loss))

def main():
    cgan = CGAN()
    cgan.train()

if __name__=='__main__':
    main()
