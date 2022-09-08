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
from utils import save_signature, test, fill_net, reset_lin_comb

#Arguments Loader
args = ArgumentParser()
max_acc = 0

#Generate Networks
train_net, test_net = ModelFactory(args)
CrossEntropy = nn.CrossEntropyLoss()
test_net = nn.DataParallel(test_net.cuda())
train_net = nn.DataParallel(train_net.cuda())
alpha = torch.zeros(args.num_alpha, requires_grad=True, device="cuda:0")

trainloader, testloader = DataLoader(args)

with torch.no_grad():
    theta = torch.cat([p.flatten() for p in train_net.parameters()])
net_optimizer = optim.SGD(train_net.parameters(), lr=1.)
lin_comb_net = torch.zeros(theta.shape).cuda()
layer_cnt = len([p for p in train_net.parameters()])
shapes = [list(p.shape) for p in train_net.parameters()]
lengths = [p.flatten().shape[0] for p in train_net.parameters()]
perm = [i for i in range(args.num_alpha)]
basis_net = torch.zeros(args.window, theta.shape[0]).cuda()
dummy_net = [torch.zeros(p.shape).cuda() for p in train_net.parameters()]
grads = torch.zeros(theta.shape, device='cuda:0')
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

lin_comb_net = reset_lin_comb(args, alpha, lin_comb_net, theta, layer_cnt, shapes, dummy_net, basis_net, lengths)
max_acc = test(train_net, test_net, lin_comb_net, testloader, lengths, shapes)
#training epochs

if args.evaluate:
    epochs = 0 

for e in range(args.epoch):
    random.shuffle(perm)
    idx = perm[:args.window]
    fill_net(args, idx, layer_cnt, shapes, dummy_net, basis_net, lengths)
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

    lin_comb_net = reset_lin_comb(args, alpha, lin_comb_net, theta, layer_cnt, shapes, dummy_net, basis_net, lengths)
    acc = test(train_net, test_net, lin_comb_net, testloader, lengths, shapes)
    if max_acc <= acc:
        max_acc = acc
        means = []
        vars = []
        for p in train_net.modules():
            if isinstance(p, nn.BatchNorm2d):
                means.append(p.running_mean)
                vars.append(p.running_var)
        if 'resnet' in args.model:
            save_signature(args, alpha, saving_path, torch.cat(means), torch.cat(vars))
        else:
            save_signature(args, alpha, saving_path)
        print("Accuracy:", acc, "Max_Accuracy:", max_acc)

if args.save_model:
    torch.save(train_net.state_dict(), "final_model.pt")

print(max_acc)

