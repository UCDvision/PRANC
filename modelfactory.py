import models
import torch.nn as nn

def ModelFactory(args):
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
    #architecture-specific setup
    if args.model == 'resnet20':
        test_net = models.resnet20(num_classes = args.num_classes)
        train_net = models.resnet20(num_classes = args.num_classes)

    if args.model == 'resnet56':
        test_net = models.resnet56(num_classes = args.num_classes)
        train_net = models.resnet56(num_classes = args.num_classes)

    if args.model == 'lenet':
        test_net = models.LeNet(num_classes = args.num_classes)
        train_net = models.LeNet(num_classes = args.num_classes)

    if args.model == 'convnet':
        test_net = models.ConvNet(3, args.num_classes, 128, args.depth, 'relu', 'instancenorm', 'avgpooling', im_size=(args.size, args.size))
        train_net = models.ConvNet(3, args.num_classes, 128, args.depth, 'relu', 'instancenorm', 'avgpooling', im_size=(args.size, args.size))

    if args.model == 'alexnet':
        test_net = models.AlexNet(num_classes=args.num_classes)
        train_net = models.AlexNet(num_classes=args.num_classes)
    
    return nn.DataParallel(train_net.cuda()), nn.DataParallel(test_net.cuda())
