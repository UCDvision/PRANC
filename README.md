# PRANC

This is the official code for paper: PRANC: Pseudo RAndom Networks for Compacting deep models. 

## Requirements:
PyTorch>=1.8

torchvision>=0.9

## Training Signature:

The single gpu mode for training signature supports cifar10, cifar100, tiny imagenet with alexnet, lenet, convnet, resnet20, and resnet56. 

```
CUDA_VISIBLE_DEVICES=<Your GPU> python3 main_1gpu.py --k 10000\
  --model resnet20 --task cifar10 --dataset path/to/dataset/ \
  --batch_size 256 --window 500 --save_path path/to/save/alphas/\
  --epoch 400 --lr 1e-3 --seed 0
```

After training the signature, for evaluating the generated alphas, you can use command:

```
CUDA_VISIBLE_DEVICES=<Your GPU> python3 main_1gpu.py --k 10000\
  --model resnet20 --task cifar10 --dataset path/to/dataset/
  --save_path path/to/saved/alphas --seed 0 --evaluate
```

For dumping the generated method:

```
CUDA_VISIBLE_DEVICES=<Your GPU> python3 main_1gpu.py --k 10000\
  --model resnet20 --task cifar10 --dataset path/to/dataset/
  --save_path path/to/saved/alphas --seed 0 --evaluate --save_model
```

## For ImageNet-100 with ResNet18:

The multi gpu mode is to support imagenet100 with larger networks such as ResNet18. Currently it only supports ResNet18 with ImageNet 100. For running multi-GPU mode:

```
CUDA_VISIBLE_DEVICES=<Your GPUs> python3 main_ngpu.py --k 20000 --window 500
```
