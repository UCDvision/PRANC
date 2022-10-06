import os
import yaml
import argparse

parser = argparse.ArgumentParser(description='Arguments of program')
parser.add_argument('config_file', metavar='F', type=str, help='Name of the config file')
args = parser.parse_args()
with open(args.config_file, 'r') as file:
    arguments = yaml.safe_load(file)

args_str = '--task_id ' + arguments['id']
args_str += ' --loss ' + arguments['experiment']['loss']
args_str += ' --method ' + arguments['experiment']['method'] 
args_str += ' --task ' + str(arguments['experiment']['task'])
args_str += ' --model ' + str(arguments['experiment']['model_arch'])
args_str += ' --img-width ' + str(arguments['dataset']['image_width'])
args_str += ' --dataset ' + str(arguments['dataset']['dataset_path']) 
args_str += ' --world_size ' + str(len(arguments['gpus']))

if arguments['experiment']['method'] == 'pranc':
    if arguments['experiment']['mode'] == 'train':
        args_str += ' --pranc_lr ' + str(arguments['experiment']['lr'])
        args_str += ' --save_path ' + str(arguments['monitor']['save_path'])
    arguments['experiment']['lr'] = 1.
    args_str += ' --seed ' + str(arguments['pranc']['seed'])
    # args_str += ' --window ' + str(arguments['pranc']['window_size'])
    args_str += ' --num_alpha ' + str(arguments['pranc']['num_alpha'])

if arguments['experiment']['scheduler'] != 'none':
    args_str += ' --scheduler ' + arguments['experiment']['scheduler']
    if arguments['experiment']['scheduler'] == 'step':
        args_str += ' --scheduler_step ' + str(arguments['experiment']['step'])
    args_str += ' --scheduler_gamma ' + str(arguments['experiment']['gamma'])

if 'weight_decay' in arguments['experiment'].keys():
    args_str += ' --weight-decay ' + str(arguments['experiment']['weight_decay'])

if 'momentum' in arguments['experiment'].keys():
    args_str += ' --momentum ' + str(arguments['experiment']['momentum'])

if arguments['experiment']['mode'] == 'train':
    args_str += ' --lr ' + str(arguments['experiment']['lr'])
    args_str += ' --epoch ' + str(arguments['experiment']['epoch'])
    args_str += ' --optimizer ' + arguments['experiment']['optimizer']
    args_str += ' --log-rate ' + str(arguments['monitor']['log_rate']) 
    args_str += ' --save_model ' + str(arguments['monitor']['save_model'])
    args_str += ' --batch_size ' + str(arguments['experiment']['batch_size'])

elif arguments['experiment']['mode'] == 'test':
    args_str += ' --lr 0'
    args_str += ' --pranc_lr 0'
    args_str += ' --epoch 0' 
    args_str += ' --batch_size 100' 
    args_str += ' --resume ' + str(arguments['experiment']['load_model'])

if 'resume' in arguments['experiment'].keys():
    args_str += ' --resume ' + str(arguments['experiment']['resume']) 

gpus = ''
for ind in arguments['gpus']:
    gpus += str(ind) + ','
print('CUDA_VISIBLE_DEVICES=' + gpus[:-1] + ' python3 main_distrib.py ' + args_str )
os.system('CUDA_VISIBLE_DEVICES=' + gpus[:-1] + ' python3 main_distrib.py ' + args_str )
