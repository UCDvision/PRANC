import models
import torch.nn as nn

def ModelFactory(args):
    if args.task == 'mnist':
        args.depth = 1
        args.num_classes = 10

    if args.task == 'cifar10':
        args.depth = 3
        args.num_classes = 10

    if args.task == 'cifar100':
        args.depth = 3
        args.num_classes = 100

    if args.task == 'tiny':
        args.depth = 4
        args.num_classes = 200
        args.size = 64

    if args.model == 'resnet20':
        train_net = models.resnet20(num_classes = args.num_classes)

    if args.model == 'resnet56':
        train_net = models.resnet56(num_classes = args.num_classes)

    if args.model == 'lenet':
        if args.task == 'mnist':
            train_net = models.LeNetMNIST(num_classes = args.num_classes)
        else:
            train_net = models.LeNet(num_classes = args.num_classes)

    if args.model == 'convnet':
        train_net = models.ConvNet(3, args.num_classes, 128, args.depth, 'relu', 'instancenorm', 'avgpooling', im_size=(args.size, args.size))

    if args.model == 'alexnet':
        train_net = models.AlexNet(num_classes=args.num_classes)
    
    return train_net
