# PRANC

This is the official code for paper: PRANC: Pseudo RAndom Networks for Compacting deep models. 

## Requirements:
PyTorch>=1.8

torchvision>=0.9

## Single GPU mode:

The single gpu mode supports cifar10, cifar100, tiny imagenet with alexnet, lenet, convnet, resnet20, and resnet56. For running in Single GPU mode:

```
CUDA_VISIBLE_DEVICES=<Your GPU> python3 main_1gpu.py --k 10000\
  --model resnet20 --task cifar10 --dataset path/to/dataset/ \
  --batch_size 256 --window 500 --save_path path/to/save/alphas/\
  --epoch 400 --lr 1e-3 --seed 0
```
This command trains 10,000 alphas for resnet20 on cifar10 for 400 epochs. It will take less than 2 hours to train 10000 alphas for 400 epochs.

## Multi GPU mode:

The multi gpu mode is to support imagenet100 with larger networks such as ResNet18. Currently it only supports ResNet18 with ImageNet 100. For running multi-GPU mode:

```
CUDA_VISIBLE_DEVICES=<Your GPUs> python3 main_ngpu.py --k 20000 --window 500
```

