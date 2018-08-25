# Pytorch implementation of AlexNet

- Now compatible with `pytorch==0.4.0`

This is an implementaiton of AlexNet, as introduced in the paper "ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky et al. ([original paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf))

This was the first very successful CNN for image classification that led to breakout of deep learning 'hype', as well as the first successful example of utilizing dropout layers.

## Prerequisites

- python >= 3.5
- pytorch==0.4.0

You can install required packages by:

```bash
pip3 install -r requirements.txt
```

## DataSet

This implemenation uses the [ILSVRC 2012 dataset](http://www.image-net.org/challenges/LSVRC/2012/), also known as the 'ImageNet 2012 dataset'.
The data size is dreadfully large (138G!), but this amount of large-sized dataset is required for successful training of AlexNet.
Testing with [Tiny ImageNet](https://tiny-imagenet.herokuapp.com/) or [MNIST](http://yann.lecun.com/exdb/mnist/) could not be done due to their smaller feature sizes (images do not fit the input size 227 x 227).

After downloading the dataset file (i.e., `ILSVRC2012_img_train.tar`), use `extract_imagenet.sh` to extract the entire dataset. 

```bash
extract_imagenet.sh
```

ImageNet 2012's dataset structure is already arranged as `/root/[class]/[img_id].jpeg`, so using `torchvision.datasets.ImageFolder` is convenient.


## Training

```bash
python3 model.py
```

Specify the data path by modifying the constant `TRAIN_IMG_DIR` at the beginning of the script.
Also tune model parameters by modifying constants at the beginning of the script.
