mkdir -p datasets
FILES=(apple2orange summer2winter_yosemite horse2zebra monet2photo cezanne2photo ukiyoe2photo vangogh2photo maps cityscapes facades iphone2dslr_flower ae_photos)
for FILE in ${FILES[*]}
do
  URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/$FILE.zip
  ZIP_FILE=./datasets/$FILE.zip
  TARGET_DIR=./datasets/$FILE/
  wget -N $URL -O $ZIP_FILE
  mkdir $TARGET_DIR
  unzip $ZIP_FILE -d ./datasets/
  rm $ZIP_FILE
done
