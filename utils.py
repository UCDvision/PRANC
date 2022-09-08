import os
import math
import torch
import torch.optim 
import torch.nn as nn
from watchdog import WatchDog


def save_signature(args, alpha, dirname, mean = None, var = None):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    torch.save(alpha, dirname + '/lr.pt')
    if 'resnet' in args.model:
        torch.save(mean, dirname + '/means.pt')
        torch.save(var, dirname + '/vars.pt')

def fill_net(args, permute, layer_cnt, shapes, dummy_net, basis_net, lengths):
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

def reset_lin_comb(args, alpha, lin_comb_net, theta, layer_cnt, shapes, dummy_net, basis_net, lengths):
    lin_comb_net = torch.zeros(theta.shape).cuda()
    start, end = 0, args.window
    while start < args.num_alpha:
        fill_net(args, range(start, end), layer_cnt, shapes, dummy_net, basis_net, lengths)
        with torch.no_grad():
            lin_comb_net += torch.matmul(basis_net.T, alpha[start:end]).T
        start = end
        end = min(end + args.window, args.num_alpha)
    return lin_comb_net

def init_alpha(args):
    alp = torch.zeros(args.num_alpha, requires_grad=True, device="cuda:0")
    with torch.no_grad():
        alp[0] = 1.
    return alp

def loss_func(args):
    return nn.CrossEntropyLoss()

def init_net(args, train_net):
    if args.seed is not None:
        torch.cuda.manual_seed(args.seed)
    for p in train_net.modules():
        if hasattr(p, 'reset_parameters'):
            p.reset_parameters()
    return train_net

def get_optimizer(args, params):
    return torch.optim.SGD(params, lr=args.lr, momentum=.9, weight_decay=1e-4)

def normal_train_single_epoch(args, epoch, train_net, trainloader, criteria, optimizer):
    train_net.train()
    train_watchdog = WatchDog()
    for batch_idx, data in enumerate(trainloader):
        train_watchdog.start()
        optimizer.zero_grad()
        imgs, labels = data
        imgs = imgs.cuda()
        labels = labels.cuda()
        loss = criteria(train_net(imgs), labels)
        loss.backward()
        optimizer.step()
        train_watchdog.stop()
        if batch_idx % args.log_rate == 0:
            print("Epoch:", epoch, "\tIteration:", batch_idx, "\tLoss:", round(loss.item(), 4), "\tTime:", train_watchdog.get_time_in_ms(), 'ms')

def save_model(args, train_net):
    if os.path.isdir(args.task_id) is False:
        os.mkdir(args.task_id)
    torch.save(train_net.state_dict(), args.task_id + '/' + args.save_model)

def load_model(args):
    return torch.load(args.resume)

def test(args, train_net, testloader):
    train_net.eval()
    cnt = 0
    total = 0
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        outputs = train_net(inputs.cuda())
        labels = labels.cuda()
        outputs = torch.argmax(outputs, dim=1)
        for i in range(outputs.shape[0]):
            if labels[i] == outputs[i]:
                cnt += 1
            total += 1

    return (cnt / total) * 100