# PRANC

This is the official code for paper `PRANC: Pseudo RAndom Networks for Compacting deep models`. PRANC is a method to compact the knowledge of a Deep Neural Net for over 50x. [Here](https://arxiv.org/abs/2206.08464) is the link to the paper.

## Requirements:
PyYAML==6.0
torch==1.10.2
torchvision==0.11.3
tqdm==4.62.3

or you can simply use:
`pip3 install -r requirements.txt`

## Running: 
For running the code, simply use:
`python3 launcher.py <config-file>`

## Config file content: 
    name: description of experiment
    id: unique identifier, your choice but be careful, will be used to store the signature and model
    gpus: list of available GPUs

    pranc.seed: seed for initializing basis networks
    num_alpha: number of basis networks

    experiments.mode: [train, test] mode of the experiment
    experiment.method: [normal, pranc] method of the experiment
    experiment.loss: [cross-entropy, mse] loss function
    experiment.lr: learning rate, can be ignored when testing
    experiment.optimizer: [sgd, adam] training optimizer
    experiment.momentum: momentum for sgd, can be ignored for other optimizers
    experiment.weight_decay: weight decay for sgd, can be ignored for other optimizers
    experiment.scheduler: [none, step, exponential], learning rate scheduler
    experiment.gamma: gamma for exponential and step scheduler 
    experiment.step: step for step scheduler
    experiment.epoch: number of training epochs
    experiment.batch_size: training batch size, optional for testing
    experiment.resume: '<TASK_ID>/pranc'  for resuming pranc training. 
    experiment.resume: '<TASK_ID>/best_model.pt'  for resuming normal training. 
    experiment.load_model: '<TASK_ID>/pranc' for pranc testing
    experiment.load_model: '<TASK_ID>/best_model.pt' for normal testing 
    experiment.task: [mnist, cifar10, cifar100, tiny] the task that is going to be solved
    experiment.model_arch: [lenet, resnet20, resnet56, alexnet, convnet] model architecture used in experiment

    dataset.image_width: input image width, set 28 for mnist, 32 for cifar, 64 tiny, 256 for imagenet
    dataset.dataset_path: path to the dataset

    monitor.log_rate: training log rate (based on batches)
    monitor.save_model: path to save the model. if touch, modify resume and load_model
    monitor.save_path: path to save the pranc signature. if touch, modify resume and load_model

There is a sample config file in `configs`


