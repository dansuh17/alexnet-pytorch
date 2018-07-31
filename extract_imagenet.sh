#!/bin/bash

mkdir imagenet

tar -xf ILSVRC2012_img_train.tar -C imagenet

cd imagenet

extract_to_folder() {
  BASENAME=$(basename $1)  # strip the path
  echo $BASENAME
  FNAME=${BASENAME%.*}   # extract filename without extension
  mkdir $FNAME
  tar -xf $1 -C $FNAME
  rm $1  # remove the tar after unzipping
}

export -f extract_to_folder
find . -name "n*.tar" -exec bash -c 'extract_to_folder {}' \;
