import os
import math
import torch
import torch.nn as nn


def save_signature(args, alpha, dirname, mean = None, var = None):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    torch.save(alpha, dirname + '/lr.pt')
    if 'resnet' in args.model:
        torch.save(mean, dirname + '/means.pt')
        torch.save(var, dirname + '/vars.pt')


def test(train_net, test_net, lin_comb_net, testloader, lengths, shapes):
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