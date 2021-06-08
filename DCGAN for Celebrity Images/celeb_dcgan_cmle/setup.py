"""Cloud ML Engine package configuration."""
from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['requests==2.19.1', 'tensorflow==1.12.0', 'matplotlib==2.2.2', 'kaggle==1.5.0', 'Pillow==8.2.0', 'urllib3==1.22']

setup(name='celeb_face_generator',
      version='4.0',
      install_requires=REQUIRED_PACKAGES,
      include_package_data=True,
      packages=find_packages(),
      description='DCGAN on Cloud ML Engine trainer on CelebA'
)
