# AlexNet-pytorch

This is an implementaiton of AlexNet, as introduced in the paper "ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky et al. ([original paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf))
It was also tested upon "Tiny ImageNet" dataset.

This was the first very successful CNN for image classification that led to breakout of deep learning 'hype', as well as the first successful example of utilizing dropout layers.

## Prerequisites

- python >= 3.5
- pytorch==0.4.0

You can install required packages by:

```bash
pip3 install -r requirements.txt
```

## DataSet

This implemenation uses "tiny imagenet" dataset, which is a smaller version of original ImageNet 2010 dataset (LSVRC-2010). Tiny ImageNet contains 200 classes instead of 1000, and has a much smaller size (237M) instaed of dreadful 138G-image data.

- [Link to ImageNet download](http://www.image-net.org/download-images) - you may need to sign up

After you retrieved the dataset and unzipping, you need to rearrange the folder structures in order to utilize pytorch's `ImageFolder`. 
This requires the data directories arranged as: `/root/[class]/[img_id].jpeg`. 
However, the newly unzipped dataset has directory structures of form: `/root/[class]/images/[img_id].jpeg`.
Included `rearrange_tiny_imagenet.py` will do the rearranging for you.

```bash
python3 rearrange_tiny_imagenet.py
```

## Training

```bash
python3 model.py
```

Specify the data path by modifying the constant `TRAIN_IMG_DIR` at the beginning of the script.
Also tune model parameters by modifying constants at the beginning of the script.
