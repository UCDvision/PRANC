import torch
from watchdog import WatchDog
from dataloader import DataLoader
from arguments import ArgumentParser
from modelfactory import ModelFactory
from utils import test, loss_func, init_net, get_optimizer, save_model, load_model, save_signature, normal_train_single_epoch, pranc_train_single_epoch, pranc_init

max_acc = 0
args = ArgumentParser()
criteria = loss_func(args)
test_watchdog = WatchDog()
train_net = ModelFactory(args)
train_net = init_net(args, train_net)
trainloader, testloader = DataLoader(args)

if args.method == 'normal':
    if args.resume is not None:
        train_net.load_state_dict(load_model(args))
        max_acc = test(args, train_net, testloader)
    
    optimizer = get_optimizer(args, train_net.parameters())
    for e in range(args.epoch):
        normal_train_single_epoch(args, e, train_net, trainloader, criteria, optimizer)
        test_watchdog.start()
        acc = test(args, train_net, testloader)
        test_watchdog.stop()
        print("TEST RESULT:\tAcc:", round(acc, 3), "\tBest Acc:", round(max_acc,3), "\tTime:", test_watchdog.get_time_in_sec(), 'seconds')
        if acc > max_acc:
            max_acc = acc
            save_model(args, train_net)
    
    acc = test(args, train_net, testloader)
    print("FINAL TEST RESULT:\tAcc:", round(max_acc, 3))

if args.method == 'pranc':
    alpha, basis_mat, init_net_weights, train_net, train_net_shape_vec = pranc_init(args, train_net)
    if args.lr > 0:
        alpha_optimizer = get_optimizer(args, [alpha], 'pranc')
        net_optimizer = get_optimizer(args, train_net.parameters(), 'network')
    if args.resume is not None:
        max_acc = test(args, train_net, testloader)
    for e in range(args.epoch):
        pranc_train_single_epoch(args, e, basis_mat, train_net, train_net_shape_vec, alpha, trainloader, criteria, alpha_optimizer, net_optimizer)
        if e % 10 == 9:
            test_watchdog.start()
            acc = test(args, train_net, testloader)
            test_watchdog.stop()
            print("TEST RESULT:\tAcc:", round(acc, 3), "\tBest Acc:", round(max_acc,3), "\tTime:", test_watchdog.get_time_in_sec(), 'seconds')
            if acc > max_acc:
                save_model(args, train_net)
                save_signature(args, alpha, train_net)
                max_acc = acc
    acc = test(args, train_net, testloader)
    print("FINAL TEST RESULT:\tAcc:", round(max_acc, 3))