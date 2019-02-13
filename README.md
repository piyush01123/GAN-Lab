

#### GAN Lab
This repository is a lab to learn about and train variants of generative adversarial networks or GANs. Most of the code is written in Keras with Tensorflow backend. There are also examples of training GANs of Cloud ML Engine.

#### Why learn GANs?
GANs are interesting because they are capable of learning even complex data distributions and generate (fake) samples from that distribution. For example:

<img src='dcgan/generated/progression.gif'/>

#### How to train GANs

The basic framework consists of 2 neural networks:
- G or Generator and
- D or Discriminator

G takes as input a noise vector and outputs a sample with same shape as real samples (we call these fake samples). D takes as input a sample (real or fake) and outputs the probability that the input sample is real. Mathematically, G and D are playing a min-max game for a value function V:

<p align="center"><img alt="$$&#10;\min_D \max_G V(D, G) = E_{x \sim X}log(D(x)) + E_{z \sim Z}log(1-D(G(z)))&#10;$$" src="svgs/676b972e9b7f5ce75615351cf625f11d.svg" align="middle" width="442.21943865pt" height="22.931502pt"/></p>

Above equation implies that D tries to minimize `V(D, G)` and G tries to maximize `V(D, G)`. Also the `log` in the above equation can be modified to things like mean-squared error as in LSGAN. The above formulation is for the binary-crossentropy loss.

In practice, we first construct G, D and AM or Adversarial Model (D stacked on top of G) and then at each training step, we teach D to differentiate between fake and real samples and simultaneously we teach AM to generate fake samples which are able to fool D. Also, D should be set to non-trainable in the 2nd part.

The idea is that with enough training, G will be able to fool D (which means we can construct real-looking samples using G.)

<!-- Ideally after training D should reach 50% accuracy and AM should reach 100% accuracy because at that point, G is able to fool D with all the samples (hence the 100% accuracy for AM) and D classifies all the real and fake samples as real (hence the 50% accuracy for D). -->

The code for the above looks like this:

```
"""Part 1 - Training discriminator(D)""
# Generate a noise of appropriate shape
noise = np.random.normal(0, 1, size=[batch_size, noise_dim])

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
# Actually, in this repo this step has been done with 2 separate train_on_batch calls for real and fake samples

"""Part 2 - Training adversarial model(G+D)"""
# Generate a noisy sample of appropriate shape
noise = np.random.normal(0, 1, size=[batch_size, noise_dim])

# Here's the trick: forcing the Adversarial model to have output class 1 for fake samples
Y = np.ones((batch_size, 1))

# Train AM (G+D) on this batch of noise and Y
a_loss = AM.train_on_batch(noise, Y)
```
Above steps are repeated for a pre-specified number of steps and hopefully

<!-- #### The GAN Class
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
Contributions and feedbacks of all kind are more than welcome; they are encouraged. -->
