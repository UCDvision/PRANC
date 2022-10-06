import os
import torch
from watchdog import WatchDog
import torch.distributed as dist
from dataloader import DataLoader
import torch.multiprocessing as mp
from arguments import ArgumentParser
from modelfactory import ModelFactory
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import test, loss_func, init_net, get_optimizer, save_model, load_model, save_signature, normal_train_single_epoch, pranc_train_single_epoch, pranc_init


def gather_all_test(gpu_ind, args, train_net, testloader):
    c, t = test(gpu_ind, args, train_net, testloader)
    total = torch.tensor([c, t], dtype=torch.float32, device=gpu_ind)
    dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
    cnt, tot = total.tolist()
    return (cnt / tot) * 100


def main_worker( gpu_ind, args):
    rank = args.global_rank + gpu_ind       
    dist.init_process_group(
    	backend='nccl',
        init_method='env://',
    	world_size=args.world_size,        
    	rank=rank
    )
    criteria = loss_func(args).to(gpu_ind)
    test_watchdog = WatchDog()
    train_net = ModelFactory(args).to(gpu_ind)
    train_net = init_net(args, train_net).to(gpu_ind)
    trainloader, testloader = DataLoader(args)
    max_acc = 0
    torch.cuda.set_device(gpu_ind)
    train_net = DDP(train_net, device_ids=[gpu_ind])
    if args.method == 'normal':
        if args.resume is not None:
            train_net.load_state_dict(load_model(gpu_ind, args))
        max_acc = gather_all_test(gpu_ind, args, train_net, testloader)
        optimizer = get_optimizer(args, train_net.parameters())
        for e in range(args.epoch):
            normal_train_single_epoch(gpu_ind, args, e, train_net, trainloader, criteria, optimizer)
            if e % 10 == 0:
                test_watchdog.start()
                acc = gather_all_test(gpu_ind, args, train_net, testloader)
                test_watchdog.stop()
                if gpu_ind == 0:
                    print("TEST RESULT:\tAcc:", round(acc, 3), "\tBest Acc:", round(max_acc,3), "\tTime:", test_watchdog.get_time_in_sec(), 'seconds')
                    if acc > max_acc:
                        max_acc = acc
                        save_model(args, train_net)
        if gpu_ind == 0: 
            print("FINAL TEST RESULT:\tAcc:", round(max_acc, 3))

    if args.method == 'pranc':
        alpha, basis_mat, init_net_weights, train_net, train_net_shape_vec = pranc_init(gpu_ind, args, train_net)
        if args.lr > 0:
            alpha_optimizer = get_optimizer(args, [alpha], 'pranc')
            net_optimizer = get_optimizer(args, train_net.parameters(), 'network')
        max_acc = gather_all_test(gpu_ind, args, train_net, testloader)
        for e in range(args.epoch):
            pranc_train_single_epoch(gpu_ind, args, e, basis_mat, train_net, train_net_shape_vec, alpha, trainloader, criteria, alpha_optimizer, net_optimizer)     #THIS IS WHERE WE ARE NOW
            if e % 10 == 9:
                test_watchdog.start()
                acc = gather_all_test(gpu_ind, args, train_net, testloader)
                test_watchdog.stop()
                print("TEST RESULT:\tAcc:", round(acc, 3), "\tBest Acc:", round(max_acc,3), "\tTime:", test_watchdog.get_time_in_sec(), 'seconds')
                if acc > max_acc:
                    save_model(args, train_net)
                    save_signature(args, alpha, train_net)              #NEEDS EDIT AS WELL
                    max_acc = acc
        print("FINAL TEST RESULT:\tAcc:", round(max_acc, 3))

if __name__ == '__main__':
    number_of_gpus = torch.cuda.device_count()
    max_acc = 0
    args = ArgumentParser()
    os.environ['MASTER_ADDR'] = args.dist_addr
    os.environ['MASTER_PORT'] = str(args.dist_port)
    if args.method == 'pranc':
        assert args.num_alpha % args.world_size == 0
    mp.spawn(main_worker, nprocs = number_of_gpus, args=(args,))
    pass