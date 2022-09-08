import time
from watchdog import WatchDog
from dataloader import DataLoader
from arguments import ArgumentParser
from modelfactory import ModelFactory
from utils import test, init_alpha, loss_func, init_net, get_optimizer, save_model, load_model, normal_train_single_epoch

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
    print("FINAL TEST RESULT:\tAcc:", round(acc, 3))

if args.method == 'pranc':
    alpha = init_alpha(args)