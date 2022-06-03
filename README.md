# PRANC

This is the official code for paper: PRANC: Pseudo RAndom Networks for Compacting deep models. 

## Single GPU mode:

The single gpu mode supports cifar10, cifar100, tiny imagenet with alexnet, lenet, convnet, resnet20, and resnet56. For running in Single GPU mode:

```
CUDA_VISIBLE_DEVICES=<Your GPU> python3 main_1gpu.py --model <model> --task <task> --dataset <dataset>
```

## Multi GPU mode:

The multi gpu mode is to support imagenet100 with larger networks such as ResNet18. Currently it only supports ResNet18 with ImageNet 100. For running multi-GPU mode:

```
CUDA_VISIBLE_DEVICES=<Your GPUs> python3 main_ngpu.py 
```

