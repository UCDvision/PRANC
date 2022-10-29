from cgi import test
import os
import torch
import torchvision
from torchvision import datasets, models
from torchvision.transforms import transforms

def DataLoader(args): 
    args.batch_size //= args.world_size
    if args.task == 'mnist':
        trainset = datasets.MNIST(args.dataset, download=True, train=True,transform=transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, sampler = train_sampler, pin_memory=True)
        
        testset = datasets.MNIST(args.dataset, download=True, train=False,transform=transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset, shuffle=False, drop_last=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,  shuffle=False, sampler=test_sampler, pin_memory=True)

        return trainloader, testloader

    if args.task == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(args.img_width, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root=args.dataset, train=True, download=True, transform=transform_train)
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker, sampler=train_sampler, pin_memory=True)

        testset = torchvision.datasets.CIFAR10(root=args.dataset, train=False, download=True, transform=transform_test)
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset, shuffle=False, drop_last=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker, sampler=test_sampler, pin_memory=True)
        return trainloader, testloader

    if args.task == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(args.img_width, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR100(root=args.dataset, train=True, download=True, transform=transform_train)
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker, sampler=train_sampler, pin_memory=True)

        testset = torchvision.datasets.CIFAR100(root=args.dataset, train=False, download=True, transform=transform_test)
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset, shuffle=False, drop_last=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker, sampler=test_sampler, pin_memory=True)
        return trainloader, testloader

    if args.task == 'tiny':
        transform_train = transforms.Compose([
            transforms.RandomCrop(args.img_width, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = datasets.ImageFolder(os.path.join(args.dataset, "train"), transform=transform_train)
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker, sampler=train_sampler, pin_memory=True)

        testset = datasets.ImageFolder(os.path.join(args.dataset, "val"), transform=transform_test)
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset, shuffle=False, drop_last=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker, sampler=test_sampler, pin_memory=True)
        return trainloader, testloader
    
    if args.task == 'imagenet' or args.task == 'imagenet100':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        transform_test =  transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        trainset = datasets.ImageFolder(os.path.join(args.dataset, "train"), transform=transform_train)
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker, sampler=train_sampler, pin_memory=True)

        testset = datasets.ImageFolder(os.path.join(args.dataset, "val"), transform=transform_test)
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset, shuffle=False, drop_last=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker, sampler=test_sampler, pin_memory=True)
        return trainloader, testloader
    
    raise "Unknown task"