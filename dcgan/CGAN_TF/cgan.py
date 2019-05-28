
"""
noise = Input(batch_size, noise_dim)
label = Input(batch_size)
fake_img = G([noise, label])
pred = D([fake_img, label])
disc_loss_fake = loss_func(pred, zeros)
pred2  = D([real_img, label])
disc_loss_real = loss_func([pred2, ones])
gen_loss = loss_func(pred, ones)
"""


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


def make_generator_model():
  """Generator.
  Returns:
    Keras Sequential model
  """
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(7*7*256, use_bias=False),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.LeakyReLU(),
      tf.keras.layers.Reshape((7, 7, 256)),
      tf.keras.layers.Conv2DTranspose(128, 5, strides=(1, 1),
                                      padding='same', use_bias=False),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.LeakyReLU(),
      tf.keras.layers.Conv2DTranspose(64, 5, strides=(2, 2),
                                      padding='same', use_bias=False),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.LeakyReLU(),
      tf.keras.layers.Conv2DTranspose(1, 5, strides=(2, 2),
                                      padding='same', use_bias=False,
                                      activation='tanh')
  ])
  return model


def make_discriminator_model():
  """Discriminator.
  Returns:
    Keras Sequential model
  """
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(64, 5, strides=(2, 2), padding='same'),
      tf.keras.layers.LeakyReLU(),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Conv2D(128, 5, strides=(2, 2), padding='same'),
      tf.keras.layers.LeakyReLU(),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(1)
  ])
  return model

def preprocess(example):
    example['image'] = tf.cast(example['image'], tf.float32)
    example['image'] = (example['image']-127.5)/127.5
    example['label'] = tf.one_hot(example['label'], 10)
    return example

class DCGAN:
    def __init__(self):
        self.batch_size = 50
        self.noise_dim=100
        self.dataset = tfds.load('mnist')['train'].map(preprocess).batch(self.batch_size)
        self.G = make_generator_model()
        self.D = make_discriminator_model()
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
        self.num_epochs = 2


    @tf.function
    def train_step(self, img_batch):
        noise = tf.random.normal([self.batch_size, self.noise_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_imgs = self.G(noise, training=True)
            
            probs_fake = self.D(fake_imgs, training=True)
            d_loss_fake = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(probs_fake), probs_fake)

            probs_real = self.D(img_batch, training=True)
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
        fake_imgs = self.G(noise, training=False)
        fig, axes = plt.subplots(4, 4)
        for i, img in enumerate(fake_imgs):
            r, c = i//4, i%4
            axes[r, c].imshow(img[:,:,0], cmap='gray')
            axes[r, c].axis('off')
        fig.savefig(self.save_dir+'%s.png' %name)

    def train(self):
        for epoch in range(self.num_epochs):
            for i, batch in enumerate(self.dataset):
                images_batch = batch['image']
                labels_batch = batch['label']
                real_X = Concatenate(batch['image'], batch['label'])
                # there are 1200 batches each 50 images
                d_loss, g_loss = self.train_step(real_X)
                if (i+1)%50==0:
                    self.plot_images("epoch_%s_batch_%s" %(epoch, i))
                    self.checkpoint.save(self.checkpoint_path)
                    template = 'Epoch {}, , Batch {}, Generator loss {}, Discriminator Loss {}'
                    print (template.format(epoch, i, g_loss, d_loss))

def main():
    dcgan = DCGAN()
    dcgan.train()

if __name__=='__main__':
    main()
