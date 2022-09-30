import os
import pdb
import math
import torch
import torch.optim 
import torch.nn as nn
from tqdm import tqdm
from watchdog import WatchDog


def save_signature(args, alpha, train_net):
    if os.path.isdir(args.task_id + '/' + args.save_path ) is False:
        os.mkdir(args.task_id + '/' + args.save_path)
    torch.save(alpha, args.task_id + '/' + args.save_path + '/alpha.pt')
    if 'resnet' in args.model:
        mean = []
        var = []
        for p in train_net.modules():
            if isinstance(p, nn.BatchNorm2d):
                mean.append(p.running_mean)
                var.append(p.running_var)
        torch.save(torch.cat(mean), args.task_id + '/' + args.save_path +  '/means.pt')
        torch.save(torch.cat(var), args.task_id + '/' + args.save_path +  '/vars.pt')

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
    print("Initializing Alpha")
    if args.resume is not None:
        alp = torch.load(args.resume + '/alpha.pt')
    else:
        alp = torch.zeros(args.num_alpha, requires_grad=True, device="cuda:0")
        with torch.no_grad():
            alp[0] = 1.
    return alp

def loss_func(args):
    if args.loss == 'mse':
        return nn.MSELoss()
    if args.loss == 'cross-entropy':
        return nn.CrossEntropyLoss()
    
def init_net(args, train_net):
    if args.seed is not None:
        print("Initializing network with seed:", args.seed)
        torch.cuda.manual_seed(args.seed)
    else:
        print("Initializing network with no seed")
    for p in train_net.modules():
        if hasattr(p, 'reset_parameters'):
            p.reset_parameters()
    return train_net

def get_optimizer(args, params, for_what='network'):
    lr = 0
    if for_what == 'network':
        lr = args.lr
    if for_what == 'pranc':
        lr = args.pranc_lr
    
    if args.optimizer == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=.9, weight_decay=1e-4)
    if args.optimizer == 'adam':
        return torch.optim.Adam(params, lr=lr)

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

def fill_basis_mat(args, train_net):
    cnt_param = sum([p.flatten().shape[0] for p in train_net.parameters()])
    this_device = 'cuda:0'
    basis_mat = torch.zeros(args.window, cnt_param, device=this_device)
    print("Initializing Basis Matrix:", list(basis_mat.shape))
    for i in tqdm(range(args.num_alpha)):
        torch.cuda.manual_seed(i)
        start_ind = 0
        for j, p in enumerate(train_net.parameters()):
            if len(p.shape) > 2:
                t = torch.zeros(p.shape, device=this_device)
                torch.nn.init.kaiming_uniform_(t, a=math.sqrt(5))
                basis_mat[i][start_ind:start_ind + t.flatten().shape[0]] = t.flatten()
                start_ind +=  t.flatten().shape[0]
                bound = 1 / math.sqrt(p.shape[1] * p.shape[2] * p.shape[3])
            if len(p.shape) == 2:
                bound = 1 / math.sqrt(p.shape[1])
                t = torch.zeros(p.shape, device=this_device)
                torch.nn.init.uniform_(t, -bound, bound)
                basis_mat[i][start_ind:start_ind + t.flatten().shape[0]] = t.flatten()
                start_ind +=  t.flatten().shape[0]
            if len(p.shape) < 2:
                t = torch.zeros(p.shape, device=this_device)
                torch.nn.init.uniform_(t , -bound, bound)
                basis_mat[i][start_ind:start_ind + t.flatten().shape[0]] = t.flatten()
                start_ind +=  t.flatten().shape[0]
    
    return basis_mat

def pranc_init(args, train_net):
    print("Initializing PRANC")
    alpha = init_alpha(args)
    basis_mat = fill_basis_mat(args, train_net)
    train_net_shape_vec = torch.zeros(basis_mat.shape[1], device=basis_mat.device)
    with torch.no_grad():
        start_ind = 0
        init_net_weights = torch.matmul(alpha, basis_mat)
        for _, p in enumerate(train_net.parameters()):
            p.copy_(init_net_weights[start_ind:start_ind + p.flatten().shape[0]].reshape(p.shape))
            start_ind +=  p.flatten().shape[0]
    if args.resume is not None:
        if 'resnet' in args.model:
            means = torch.load(args.resume + '/means.pt')
            vars = torch.load(args.resume + '/vars.pt')
        ind = 0
        for p1 in train_net.modules():
            if isinstance(p1, nn.BatchNorm2d):
                leng = p1.running_var.shape[0]
                p1.running_mean.copy_(means[ind:ind + leng])
                p1.running_var.copy_(vars[ind:ind + leng])
                ind += leng
    return alpha, basis_mat, init_net_weights, train_net, train_net_shape_vec

def get_train_net_grads(train_net, train_net_grad_vec):
    with torch.no_grad():
        start_ind = 0
        for p in train_net.parameters():
            length = p.flatten().shape[0]
            train_net_grad_vec[start_ind:start_ind + length] = p.grad.flatten()
            start_ind += length
        return train_net_grad_vec

def update_train_net(alpha, basis_mat, train_net, train_net_shape_vec):
    train_net_shape_vec = torch.matmul(alpha, basis_mat)
    with torch.no_grad():
        start_ind = 0
        for p in train_net.parameters():
            length = p.flatten().shape[0]
            p.copy_(train_net_shape_vec[start_ind: start_ind + length].reshape(p.shape))
            start_ind += length
        return train_net

def pranc_train_single_epoch(args, epoch, basis_mat, train_net, train_net_shape_vec, alpha, trainloader, criteria, alpha_optimizer, net_optimizer):
    train_net.train()
    train_watchdog = WatchDog()
    for batch_idx, data in enumerate(trainloader):
        train_watchdog.start()
        net_optimizer.zero_grad()
        alpha_optimizer.zero_grad()
        imgs, labels = data
        imgs = imgs.cuda()
        labels = labels.cuda()
        loss = criteria(train_net(imgs), labels)
        loss.backward()
        train_net_shape_vec = get_train_net_grads(train_net, train_net_shape_vec)
        alpha.grad = torch.matmul(train_net_shape_vec, basis_mat.T)
        alpha_optimizer.step()
        train_net = update_train_net(alpha, basis_mat, train_net, train_net_shape_vec)
        train_watchdog.stop()
        if batch_idx % args.log_rate == 0:
            print("Epoch:", epoch, "\tIteration:", batch_idx, "\tLoss:", round(loss.item(), 4), "\tTime:", train_watchdog.get_time_in_ms(), 'ms')

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