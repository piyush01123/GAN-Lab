"""Cloud ML Engine package configuration."""
from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['requests==2.19.1', 'tensorflow==2.5.1', 'matplotlib==2.2.2']

setup(name='mnist_generator',
      version='4.0',
      install_requires=REQUIRED_PACKAGES,
      include_package_data=True,
      packages=find_packages(),
      description='MNIST GAN on Cloud ML Engine'
)
