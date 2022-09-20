import argparse


def ArgumentParser():
    parser = argparse.ArgumentParser(description='Arguments of program')
    
    parser.add_argument('--dataset', required=True, type=str, help='Path to dataset')
    parser.add_argument('--num_alpha', default=None, type=int, help='Number of alphas')
    parser.add_argument('--lr', required=True, type=float, help='Learning rate for network')
    parser.add_argument('--resume', default=None, type=str, help='Resume previous training')
    parser.add_argument('--epoch', required=True, type=int, help='Number of training epochs')
    parser.add_argument('--seed', default=None, type=int, help='Seed to initialize networks')
    parser.add_argument('--save_path', default=None, type=str, help='Path to save the alphas')
    parser.add_argument('--save_model', default=None, type=str, help='Path to save the model')
    parser.add_argument('--pranc_lr', default=None, type=float, help='Learning rate for alpha')
    parser.add_argument('--img-width', required=True, type=int, help='Width of the input image')
    parser.add_argument('--batch_size', required=True, type=int, help='Data batch size for training')
    parser.add_argument('--method', required=True, type=str, help='How to train model ["pranc", "normal"]')
    parser.add_argument('--log-rate', default=None, type=int, help='Rate of logging the training progress')
    parser.add_argument('--task_id', required=True, type=str, help='Experiment ID, Used for saving the model')
    parser.add_argument('--task', required=True, type=str, help='Classification task to solve [cifar10, cifar100, tiny]')
    parser.add_argument('--window', default=None, type=int, help='The number of alphas selected in each coordinate descent')
    parser.add_argument('--model', required=True, type=str, help='Architecture to train the PRANC [lenet, alexnet, resnet20, resnet56, convnet]')

    
    return parser.parse_args()
