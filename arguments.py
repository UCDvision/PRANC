import argparse


def ArgumentParser():
    parser = argparse.ArgumentParser(description='Arguments of program')
    parser.add_argument('--epoch', default=1, type=int, help='Number of training epochs (Default: 1)')
    parser.add_argument('--seed', default=0, type=int, help='Seed to initialize networks (Default: 0)')
    parser.add_argument('--num_alpha', default=10000, type=int, help='Number of alphas (Default: 10,000)')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate for alpha (Default: 0.001)')
    parser.add_argument('--img-width', default=32, type=int, help='Width of the input image (Default: 32)')
    parser.add_argument('--batch_size', default=256, type=int, help='Data batch size for training (Default: 256)')
    parser.add_argument('--resume', action='store_true', help='Resume previous training (Add the flag to set True)')
    parser.add_argument('--dataset', default='./datasets', type=str, help='Path to dataset (Default: ./datasets)')
    parser.add_argument('--save_model', action='store_true', help='Save the model or not (Add the flag to set True)')
    parser.add_argument('--log-rate', default=50, type=int, help='Rate of logging the training progress (Default: 50)')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate a pretrained PRANC model (Add the flag to set True)')
    parser.add_argument('--save_path', default='./random_basis', type=str, help='Path to save the alphas (Default: ./random_basis)')
    parser.add_argument('--window', default=500, type=int, help='The number of alphas selected in each coordinate descent (Default: 500)')
    parser.add_argument('--task', default='cifar10', type=str, help='Classification task to solve [cifar10, cifar100, tiny] (Default: cifar10)')
    parser.add_argument('--model', default='resnet20', type=str, help='Architecture to train the PRANC [lenet, alexnet, resnet20, resnet56, convnet] (Default: resnet20)')

    return parser.parse_args()
