import os
import math
import time
import torch
import random
import models
import argparse
import torchvision
from torch import nn
from torch import optim
from torchvision import datasets
from torchvision.transforms import transforms
#Arguments
parser = argparse.ArgumentParser(description='Arguments of program')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--size', default=32, type=int)
parser.add_argument('--k', default=10000, type=int)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--save_model', action='store_true')
parser.add_argument('--epoch', default=1, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--window', default=500, type=int, help="The number of alphas in each coordinate descent (the m in paper)")
parser.add_argument('--log-rate', default=50, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--dataset', default='../datasets', type=str)
parser.add_argument('--save_path', default='./random_basis', type=str)
parser.add_argument('--task', default='cifar10', type=str, help='options: cifar10, cifar100, tiny')
parser.add_argument('--model', default='resnet20', type=str, help='options: lenet, alexnet, resnet20, resnet56, convnet')
args = parser.parse_args()
#basic setups
n = args.k
max_acc = 0
test_interval = 1
epochs = args.epoch
window = args.window
log_rate = args.log_rate
batch_size = args.batch_size

#task-specific setup
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

CrossEntropy = nn.CrossEntropyLoss()
test_net = nn.DataParallel(test_net.cuda())
train_net = nn.DataParallel(train_net.cuda())
alpha = torch.zeros(args.k, requires_grad=True, device="cuda:0")
#dataloaders and augmentations
if args.task == 'cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(args.size, padding=4),
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=32)

    testset = torchvision.datasets.CIFAR10(root=args.dataset, train=False, download=True, transform=transform_test)


if args.task == 'cifar100':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=32)

    testset = torchvision.datasets.CIFAR100(root=args.dataset, train=False, download=True, transform=transform_test)


if args.task == 'tiny':
    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=32)

    testset = datasets.ImageFolder(os.path.join(args.dataset, "val"), transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=32)


def save_signature(dirname, mean = None, var = None):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    torch.save(alpha, dirname + '/lr.pt')
    if 'resnet' in args.model:
        torch.save(mean, dirname + '/means.pt')
        torch.save(var, dirname + '/vars.pt')


with torch.no_grad():
    theta = torch.cat([p.flatten() for p in train_net.parameters()])
net_optimizer = optim.SGD(train_net.parameters(), lr=1.)
lin_comb_net = torch.zeros(theta.shape).cuda()
layer_cnt = len([p for p in train_net.parameters()])
shapes = [list(p.shape) for p in train_net.parameters()]
lengths = [p.flatten().shape[0] for p in train_net.parameters()]

#evaluation function
def test():
    with torch.no_grad():
        start_ind = 0
        for j, p in enumerate(test_net.parameters()):
            p.copy_(lin_comb_net[start_ind:start_ind + lengths[j]].view(shapes[j]))
            start_ind += lengths[j]
        for p1, p2 in zip(test_net.modules(), train_net.modules()):
            if isinstance(p1, nn.BatchNorm2d):
                p1.running_mean.copy_(p2.running_mean)
                p1.running_var.copy_(p2.running_var)

    cnt = 0
    total = 0
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=32)

    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        outputs = test_net(inputs.cuda())
        labels = labels.cuda()
        outputs = torch.argmax(outputs, dim=1)
        for i in range(outputs.shape[0]):
            if labels[i] == outputs[i]:
                cnt += 1
            total += 1

    return (cnt / total) * 100

perm = [i for i in range(n)]
basis_net = torch.zeros(window, theta.shape[0]).cuda()
dummy_net = [torch.zeros(p.shape).cuda() for p in train_net.parameters()]
grads = torch.zeros(theta.shape, device='cuda:0')

#initializing basis networks
def fill_net(permute):
    bound = 1
    for j, p in enumerate(permute):
        torch.cuda.manual_seed_all(p + n * args.seed)
        start_ind = 0
        for i in range(layer_cnt):
            if len(shapes[i]) > 2:
                torch.nn.init.kaiming_uniform_(dummy_net[i], a=math.sqrt(5))
                basis_net[j][start_ind:start_ind + lengths[i]] = dummy_net[i].flatten()
                start_ind += lengths[i]
                bound = 1 / math.sqrt(shapes[i][1] * shapes[i][2] * shapes[i][3])
            if len(shapes[i]) == 2:
                bound = 1 / math.sqrt(shapes[i][1])
                torch.nn.init.uniform_(dummy_net[i], -bound, bound)
                basis_net[j][start_ind:start_ind + lengths[i]] = dummy_net[i].flatten()
                start_ind += lengths[i]
            if len(shapes[i]) < 2:
                torch.nn.init.uniform_(dummy_net[i], -bound, bound)
                basis_net[j][start_ind:start_ind + lengths[i]] = dummy_net[i].flatten()
                start_ind += lengths[i]


saving_path = args.save_path + '_' + args.task + '_' + args.model + '_' + str(args.k)
if args.resume:
    with torch.no_grad():
        alpha = torch.load(saving_path + '/lr.pt').cuda()
        if 'resnet' in args.model:
            means = torch.load(saving_path + '/means.pt')
            vars = torch.load(saving_path + '/vars.pt')
        ind = 0
        for p1 in train_net.modules():
            if isinstance(p1, nn.BatchNorm2d):
                leng = p1.running_var.shape[0]
                p1.running_mean.copy_(means[ind:ind + leng])
                p1.running_var.copy_(vars[ind:ind + leng])
                ind += leng
else:
    with torch.no_grad():
        alpha[0] = 1.
#calculating linear combination of basis networks and alphas
def reset_lin_comb():
    global lin_comb_net
    lin_comb_net = torch.zeros(theta.shape).cuda()
    start, end = 0, window
    while start < n:
        fill_net(range(start, end))
        with torch.no_grad():
            lin_comb_net += torch.matmul(basis_net.T, alpha[start:end]).T
        start = end
        end = min(end + window, n)

reset_lin_comb()
max_acc = test()
#training epochs

if args.evaluate:
    epochs = 0 

for e in range(epochs):
    random.shuffle(perm)
    idx = perm[:window]
    fill_net(idx)
    with torch.no_grad():
        rest_of_net = lin_comb_net - torch.matmul(basis_net.T, alpha[idx]).T
    optimizer = torch.optim.SGD([alpha], lr=args.lr, momentum=.9, weight_decay=1e-4)
    for i, data in enumerate(trainloader):
        optimizer.zero_grad()
        net_optimizer.zero_grad()
        imgs, labels = data
        imgs = imgs.cuda()
        labels = labels.cuda()
        select_subnet = torch.matmul(basis_net.T, alpha[idx]).T
        with torch.no_grad():
            start_ind = 0
            for j, p in enumerate(train_net.parameters()):
                p.copy_((select_subnet + rest_of_net)[start_ind:start_ind + lengths[j]].view(shapes[j]))
                start_ind += lengths[j]

        loss = CrossEntropy(train_net(imgs), labels)
        if i % log_rate == 0:
            print("Epoch:", e, "\tIteration:", i, "\tLoss:", loss.item())
        loss.backward()
        with torch.no_grad():
            start_ind = 0
            for j, p in enumerate(train_net.parameters()):
                grads[start_ind:start_ind + lengths[j]].copy_(p.grad.flatten())
                start_ind += lengths[j]
        if alpha.grad is None:
            alpha.grad = torch.zeros(alpha.shape, device=alpha.get_device())
        alpha.grad[idx] = torch.matmul(grads, basis_net.T)
        optimizer.step()

    with torch.no_grad():
        lin_comb_net.copy_(rest_of_net + torch.matmul(basis_net.T, alpha[idx]).T)
    if e % test_interval == test_interval - 1:
        reset_lin_comb()
        acc = test()
        if max_acc <= acc:
            max_acc = acc
            means = []
            vars = []
            for p in train_net.modules():
                if isinstance(p, nn.BatchNorm2d):
                    means.append(p.running_mean)
                    vars.append(p.running_var)
            if 'resnet' in args.model:
                save_signature(saving_path, torch.cat(means), torch.cat(vars))
            else:
                save_signature(saving_path)
        print("Accuracy:", acc, "Max_Accuracy:", max_acc)

if args.save_model:
    torch.save(train_net.state_dict(), "final_model.pt")

print(max_acc)

