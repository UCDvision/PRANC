import os
import math
import pdb
import threading
import time
import torch
import random
import argparse
import torchvision
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, models
from torchvision.transforms import transforms


parser = argparse.ArgumentParser(description='Arguments of program')
parser.add_argument('--k', default=20000, type=int)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--epoch', default=400, type=int)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--window', default=500, type=int, help="The number of alphas in each coordinate descent (the M in paper)")
parser.add_argument('--log-rate', default=1, type=int)
parser.add_argument('--dataset', default="/datasets/imagenet100", type=str)
args = parser.parse_args()
n = args.k
max_acc = 0
batch_size = 256
epochs = args.epoch
window = args.window
log_rate = args.log_rate
CE = nn.CrossEntropyLoss()
test_interval = 1
saving_path = './r18_batch_imagenet100_' + str(n)
alpha = torch.zeros(n, requires_grad=True, device="cuda:0")
if alpha.grad is None:
    alpha.grad = torch.zeros(alpha.shape, device=alpha.get_device())


transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = datasets.ImageFolder(os.path.join(args.dataset, "train"), transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=64, pin_memory=True)
testset = datasets.ImageFolder(os.path.join(args.dataset, "val"), transform=transform_test)


def save_signature(dirname, mean, var):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    torch.save(alpha, dirname + '/lr.pt')
    torch.save(mean, dirname + '/means.pt')
    torch.save(var, dirname + '/vars.pt')


num_gpus = torch.cuda.device_count()
train_net = nn.DataParallel(models.resnet18(num_classes=100).cuda())
train_net.train()
net_optimizer = optim.SGD(train_net.parameters(), lr=1.)
with torch.no_grad():
    theta = torch.cat([p.flatten() for p in train_net.parameters()])
lin_comb_net = torch.zeros(theta.shape).cuda()
layer_cnt = len([p for p in train_net.parameters()])
shapes = [list(p.shape) for p in train_net.parameters()]
lengths = [p.flatten().shape[0] for p in train_net.parameters()]
test_net = nn.DataParallel(models.resnet18(num_classes=100).cuda())
test_net.eval()


def test():
    test_cnt, test_total = 0, 0
    with torch.no_grad():
        start_ind = 0
        for j, p in enumerate(test_net.parameters()):
            p.copy_(lin_comb_net[start_ind:start_ind + lengths[j]].view(shapes[j]))
            start_ind += lengths[j]
        for p1, p2 in zip(test_net.modules(), train_net.modules()):
            if isinstance(p1, nn.BatchNorm2d):
                p1.running_mean.copy_(p2.running_mean)
                p1.running_var.copy_(p2.running_var)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=64, pin_memory=True)
    for i, data in enumerate(testloader):
        inputs, lbls = data
        inputs = inputs.cuda()
        lbls = lbls.cuda()
        outputs = test_net(inputs)
        outputs = torch.argmax(outputs, dim=1)
        for i in range(outputs.shape[0]):
            if lbls[i] == outputs[i]:
                test_cnt += 1
            test_total += 1

    return (test_cnt / test_total) * 100


perm = [i for i in range(n)]
basis_net = [torch.zeros(window // num_gpus, theta.shape[0]).to(i) for i in range(num_gpus)]
dummy_net = [[torch.zeros(p.shape).to(i) for p in train_net.parameters()] for i in range(num_gpus)]
grads = torch.zeros(theta.shape).cuda()
channels = torch.zeros(num_gpus, theta.shape[0]).to(0)
fill_net_sem = threading.Semaphore(1)


def fill_net_(permute, gpu_ind):
    boundary = 1
    for j, p in enumerate(permute):
        fill_net_sem.acquire()
        torch.cuda.set_device(gpu_ind)
        torch.cuda.manual_seed(p)
        start_ind = 0
        for i in range(layer_cnt):
            if len(shapes[i]) > 2:
                torch.cuda.set_device(gpu_ind)
                torch.nn.init.kaiming_uniform_(dummy_net[gpu_ind][i], a=math.sqrt(5))
                basis_net[gpu_ind][j][start_ind:start_ind + lengths[i]] = dummy_net[gpu_ind][i].flatten()
                start_ind += lengths[i]
                boundary = 1 / math.sqrt(shapes[i][1] * shapes[i][2] * shapes[i][3])
            if len(shapes[i]) == 2:
                boundary = 1 / math.sqrt(shapes[i][1])
                torch.cuda.set_device(gpu_ind)
                torch.nn.init.uniform_(dummy_net[gpu_ind][i], -boundary, boundary)
                basis_net[gpu_ind][j][start_ind:start_ind + lengths[i]] = dummy_net[gpu_ind][i].flatten()
                start_ind += lengths[i]
            if len(shapes[i]) < 2:
                torch.cuda.set_device(gpu_ind)
                torch.nn.init.uniform_(dummy_net[gpu_ind][i], -boundary, boundary)
                basis_net[gpu_ind][j][start_ind:start_ind + lengths[i]] = dummy_net[gpu_ind][i].flatten()
                start_ind += lengths[i]
        fill_net_sem.release()


if args.resume:
    with torch.no_grad():
        alpha = torch.load(saving_path + '/lr.pt').cuda()
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


def fill_nets(permute, gpu_ind):
    fill_net_(permute, gpu_ind)
    channels[gpu_ind] = torch.matmul(basis_net[gpu_ind][:len(permute)].T, alpha[permute].to(gpu_ind)).T.to(0)


def fill_net(permute):
    interval = window // num_gpus
    threads = []
    for i in range(num_gpus):
        threads.append(threading.Thread(target=fill_nets, args=(permute[i * interval: (i+1) * interval], i)))
        threads[-1].start()
    for t in threads:
        t.join()


def reset_lin_comb():
    global lin_comb_net
    lin_comb_net = torch.zeros(theta.shape).cuda()
    start, end = 0, window
    with torch.no_grad():
        while start < n:
            fill_net(range(start, end))
            lin_comb_net += torch.sum(channels, dim=0)
            start = end
            end = min(end + window, n)

reset_lin_comb()
print("Initial Testing")
max_acc = test()
print("Current Accuracy:", max_acc)


def single_gpu_matmul(gpu_ind, perm):
    channels[gpu_ind] = torch.matmul(basis_net[gpu_ind][:len(perm)].T, alpha[perm].to(gpu_ind)).T.to(0)


def distrib_matmul(perm):
    threads = []
    interval = window // num_gpus
    for i in range(num_gpus):
        if i * interval >= len(perm):
            break
        threads.append(threading.Thread(target=single_gpu_matmul, args=(i, perm[i * interval: min((i + 1) * interval, len(perm))])))
        threads[-1].start()
    for t in threads:
        t.join()
    return torch.sum(channels, dim=0)


for e in range(epochs):
    random.shuffle(perm)
    idx = perm[:window]
    fill_net(idx)
    with torch.no_grad():
        rest_of_net = lin_comb_net - distrib_matmul(idx)
    optimizer = torch.optim.Adam([alpha], lr=args.lr)
    for i, data in enumerate(trainloader):
        optimizer.zero_grad()
        net_optimizer.zero_grad()
        imgs, labels = data
        imgs = imgs.cuda()
        labels = labels.cuda()
        select_subnet = distrib_matmul(idx)
        with torch.no_grad():
            start_ind = 0
            for j, p in enumerate(train_net.parameters()):
                p.copy_((select_subnet + rest_of_net)[start_ind:start_ind + lengths[j]].view(shapes[j]))
                start_ind += lengths[j]

        loss = CE(train_net(imgs), labels)
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
        for gpu in range(num_gpus):
            start, end = gpu * window // num_gpus , (gpu + 1) * window // num_gpus
            alpha.grad[idx[start:end]] = torch.matmul(grads.to(gpu), basis_net[gpu][:window].T).to(0)
        optimizer.step()

    reset_lin_comb()
    if e % test_interval == test_interval - 1:
        acc = test()
        if max_acc < acc:
            max_acc = acc
            means = []
            vars = []
            for p in train_net.modules():
                if isinstance(p, nn.BatchNorm2d):
                    means.append(p.running_mean)
                    vars.append(p.running_var)
            save_signature(saving_path, torch.cat(means), torch.cat(vars))
        print("Accuracy:", acc, "Max_Accuracy:", max_acc)


print(max_acc)
