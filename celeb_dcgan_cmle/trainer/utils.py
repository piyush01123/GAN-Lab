import os

def download_dataset():
    os.system('export KAGGLE_USERNAME=datadinosaur'
    os.system('export KAGGLE_KEY=xxxxxxxxxxxxxx')
    os.system('kaggle datasets download -d jessicali9530/celeba-dataset'))

def copy_to_gcs(local_file, gcs_file):
    os.system('gsutil cp %s %s' %(local_file, gcs_file))
