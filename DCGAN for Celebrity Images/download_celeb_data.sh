
export KAGGLE_USERNAME='piyushkgp'
export KAGGLE_KEY='3aa35c9a60b2289e66d283d182514e78'
kaggle datasets download -d jessicali9530/celeba-dataset
unzip celeba-dataset.zip datasets/
unzip datasets/*.zip datasets/
rm datasets/*.zip
rm *.zip
