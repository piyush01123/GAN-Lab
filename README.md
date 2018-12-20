#### GAN Lab
This repository is a lab to create and train variants of generative adversarial networks or GANs. All models are written in Keras with Tensorflow backend.

#### Training GANs
This is how we train a GAN:
Let's say we have a generator G and a discriminator D.
For example in `example.py` we have a G which maps tensors of shape (None, 100) to (None, 28, 28, 1) and D which maps tensors of shape (None, 28, 28, 1) to (None, 1)
G takes as input a noise vector and outputs a sample with same shape as real samples.
D takes as input a sample and outputs the probability that the input sample is real.

Then at each training step, we are teaching D to differentiate between fake and real samples and then AM (G+D) to create fake samples which are able to fool D.

The code for this looks like this:

```
# Teaching D to tell fake from real samples
noise = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.noise_dim])
fake = self.G.predict(noise)
real = self.data[np.random.randint(0, self.data.shape[0], self.batch_size), :, :, :]
train_x = np.concatenate((real, fake))
train_y = np.concatenate((np.ones((self.batch_size, 1)), np.zeros((self.batch_size, 1))))
d_loss = self.D.train_on_batch(train_x, train_y)

# Teaching AM or G+D to create fake samples which are able to fool D.
noise = np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.noise_dim])
Y = np.ones((self.batch_size, 1)) # we are forcing the Adversarial model to have output class 1 for fake samples
a_loss = self.AM.train_on_batch(noise, Y)
```

#### GAN Class
I have created a `GAN` class in `gan.py` which takes in its constructor, a generator and a discriminator as arguments.
Both of these need to be Keras' `Model` objects
It contains methods `makeGAN` and `train` which can be called on a `GAN` object to create GAN from G and D and then train it.
I have included an example in `example.py` to use this class.
