import argparse


def ArgumentParser():
    parser = argparse.ArgumentParser(description='Arguments of program')
    
    parser.add_argument('--world_size', default=4, type=int, help='Number of GPUs')
    parser.add_argument('--dataset', required=True, type=str, help='Path to dataset')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum of SGD')
    parser.add_argument('--num_alpha', default=None, type=int, help='Number of alphas')
    parser.add_argument('--global_rank', default=0, type=int, help='Rank among all GPUs')
    parser.add_argument('--dist_port', default=8880, type=int, help='Master Port of access')
    parser.add_argument('--lr', required=True, type=float, help='Learning rate for network')
    parser.add_argument('--resume', default=None, type=str, help='Resume previous training')
    parser.add_argument('--epoch', required=True, type=int, help='Number of training epochs')
    parser.add_argument('--seed', default=0, type=int, help='Seed to initialize networks')
    parser.add_argument('--scheduler', default='none', type=str, help='Training lr Scheduler')
    parser.add_argument('--save_path', default=None, type=str, help='Path to save the alphas')
    parser.add_argument('--save_model', default=None, type=str, help='Path to save the model')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='Weight Decay of SGD')
    parser.add_argument('--pranc_lr', default=None, type=float, help='Learning rate for alpha')
    parser.add_argument('--optimizer', default=None, type=str, help='Optimizer of the network')
    parser.add_argument('--dist_addr', default='localhost', type=str, help='Address of Master')
    parser.add_argument('--img-width', required=True, type=int, help='Width of the input image')
    parser.add_argument('--loss', required=True, type=str, help='Loss function for the experiment')
    parser.add_argument('--batch_size', required=True, type=int, help='Data batch size for training')
    parser.add_argument('--num_worker', default=8, type=int, help='Number of workers for dataloaders')
    parser.add_argument('--method', required=True, type=str, help='How to train model ["pranc", "normal"]')
    parser.add_argument('--log-rate', default=None, type=int, help='Rate of logging the training progress')
    parser.add_argument('--task_id', required=True, type=str, help='Experiment ID, Used for saving the model')
    parser.add_argument('--scheduler_step', default=0, type=int, help='Steps for scheduler, only for step scheduler')
    parser.add_argument('--task', required=True, type=str, help='Classification task to solve [cifar10, cifar100, tiny]')
    parser.add_argument('--window', default=None, type=int, help='The number of alphas selected in each coordinate descent')
    parser.add_argument('--scheduler_gamma', default=0.99, type=float, help='Gamma for updating learning rate in scheduler')
    parser.add_argument('--model', required=True, type=str, help='Architecture to train the PRANC [lenet, alexnet, resnet20, resnet56, convnet]')

    
    return parser.parse_args()
