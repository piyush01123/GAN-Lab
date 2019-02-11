import os
from . import cred

def download_dataset():
    os.environ['KAGGLE_USERNAME'] = cred.KAGGLE_USERNAME
    os.environ['KAGGLE_KEY'] =  cred.KAGGLE_KEY
    os.system('kaggle datasets download -d jessicali9530/celeba-dataset')
    os.system('unzip celeba-dataset.zip')

def copy_to_gcs(local_file, gcs_file):
    os.system('gsutil cp %s %s' %(local_file, gcs_file))
