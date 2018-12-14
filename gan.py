from keras.models import Sequential
from keras.layers import

class GAN:
    def Discriminaor(self):
        D = Sequential()
        return D
    def Generator(self):
        G = Sequential()
        return G
    def makeGAN(self):
        pass
    def train(self):
        gan = self.makeGAN()
        gan.fit()

if __name__ == '__main__':
    gan = GAN()
    gan.train()
