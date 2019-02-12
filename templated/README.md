#### GAN Lab
This repository is a lab to create and train variants of generative adversarial networks or GANs. All models are written in Keras with Tensorflow backend.

#### Training GANs
This is how we train a GAN:

Let's say we have a generator G and a discriminator D.
For example in `example.py` we have a G which maps tensors of shape (None, 100) to (None, 28, 28, 1) and D which maps tensors of shape (None, 28, 28, 1) to (None, 1).

G takes as input a noise vector and outputs a sample with same shape as real samples (we call these fake samples).
D takes as input a sample (real or fake) and outputs the probability that the input sample is real.

Then at each training step, we teach D to differentiate between fake and real samples and then AM (G+D) to create fake samples which are able to fool D.

The idea is that with enough training, G will be able to fool D (which means we can construct real-looking samples using G.)

The code for the above looks like this:

```
"""Part 1 - Training discriminator(D)""
# Generate a noise of appropriate shape
noise = np.random.uniform(-1.0, 1.0, size=[batch_size, noise_dim])

# Generate a fake sample from this random noise
fake = G.predict(noise)

# Take a sample from real data
real = data[np.random.randint(0, data.shape[0], batch_size), :, :, :]

# train_x is the concatenation of real and fake samples
train_x = np.concatenate((real, fake))

# and train_y is the matrix containing 1's for real and 0's for fake samples
train_y = np.concatenate((np.ones((batch_size, 1)), np.zeros((batch_size, 1))))

# Train D on this batch of train_x and train_y
d_loss = D.train_on_batch(train_x, train_y)

"""Part 2 - Training adversarial model(G+D)"""
# Generate a noisy sample of appropriate shape
noise = np.random.uniform(-1.0, 1.0, size=[batch_size, noise_dim])

# Here's the trick: forcing the Adversarial model to have output class 1 for fake samples
Y = np.ones((batch_size, 1))

# Train AM (G+D) on this batch of noise and Y
a_loss = AM.train_on_batch(noise, Y)
```
Note that both of these losses are minimized simultaneously which means the discriminator and the generator are getting trained together.

#### The GAN Class
I have created a `GAN` class in `gan.py` which takes in its constructor, a generator and a discriminator as arguments.
Both of these need to be Keras' `Model` objects
It contains methods `makeGAN` and `train` which can be called on a `GAN` object to create GAN from G and D and then train it.
The API for this class is not direct though and it must be defined by inheritance.
For example you would use this class in this way:
```
from gan import GAN
class demoGAN(GAN):
    def __init__(self, kwargs):
        self.G = self.Generator()
        self.D = self.Discriminator()
        super().__init__(kwargs)
    def Generator(self):
        # Generator model goes here
        pass
    def Discriminator(self):
        # Discriminator model goes here
        pass

demo_gan = demoGAN()
demo_gan.makeGAN()
demo_gan.train()
```
I have included a complete example in `example.py` to use this class.
My hope is that this class will make experimentation much faster because now new `GAN` objects can inherit all the boilerplate methods for making and training GANs from the `GAN` class.
Contributions and feedbacks of all kind are more than welcome; they are encouraged.
