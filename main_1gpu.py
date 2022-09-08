import os
import math
import time
import torch
import random
from torch import nn
from torch import optim
from torchvision import datasets
from dataloader import DataLoader
from arguments import ArgumentParser
from modelfactory import ModelFactory
from torchvision.transforms import transforms

#Arguments Loader
args = ArgumentParser()
max_acc = 0

#Generate Networks
train_net, test_net = ModelFactory(args)
CrossEntropy = nn.CrossEntropyLoss()
test_net = nn.DataParallel(test_net.cuda())
train_net = nn.DataParallel(train_net.cuda())
alpha = torch.zeros(args.num_alpha, requires_grad=True, device="cuda:0")

#dataloaders and augmentations
trainloader, testloader = DataLoader(args)

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

perm = [i for i in range(args.num_alpha)]
basis_net = torch.zeros(args.window, theta.shape[0]).cuda()
dummy_net = [torch.zeros(p.shape).cuda() for p in train_net.parameters()]
grads = torch.zeros(theta.shape, device='cuda:0')

#initializing basis networks
def fill_net(permute):
    bound = 1
    for j, p in enumerate(permute):
        torch.cuda.manual_seed_all(p + args.num_alpha * args.seed)
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


saving_path = args.save_path + '_' + args.task + '_' + args.model + '_' + str(args.num_alpha)
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
    start, end = 0, args.window
    while start < args.num_alpha:
        fill_net(range(start, end))
        with torch.no_grad():
            lin_comb_net += torch.matmul(basis_net.T, alpha[start:end]).T
        start = end
        end = min(end + args.window, args.num_alpha)

reset_lin_comb()
max_acc = test()
#training epochs

if args.evaluate:
    epochs = 0 

for e in range(args.epoch):
    random.shuffle(perm)
    idx = perm[:args.window]
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
        if i % args.log_rate == 0:
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

