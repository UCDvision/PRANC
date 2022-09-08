import os
import yaml
import argparse

parser = argparse.ArgumentParser(description='Arguments of program')
parser.add_argument('config_file', metavar='F', type=str, help='Name of the config file')
args = parser.parse_args()
with open(args.config_file, 'r') as file:
    arguments = yaml.safe_load(file)

args_str = '--epoch ' + str(arguments['experiment']['epoch']) + \
            ' --seed ' + str(arguments['pranc']['seed']) + \
            ' --num_alpha ' + str(arguments['pranc']['num_alpha']) + \
            ' --lr ' + str(arguments['experiment']['lr']) + \
            ' --img-width ' + str(arguments['dataset']['image_width']) + \
            ' --batch_size ' + str(arguments['experiment']['batch_size']) + \
            ' --resume ' + str(arguments['experiment']['resume']) + \
            ' --dataset ' + str(arguments['dataset']['dataset_path']) + \
            ' --save_model ' + str(arguments['monitor']['save_model']) + \
            ' --log-rate ' + str(arguments['monitor']['log_rate']) + \
            ' --save_path ' + str(arguments['monitor']['save_path']) + \
            ' --window ' + str(arguments['experiment']['window_size']) + \
            ' --task ' + str(arguments['experiment']['task']) + \
            ' --model ' + str(arguments['experiment']['model_arch']) + \
            ' --evaluate ' + 'False' if arguments['experiment']['mode'] == 'train' else 'True'

print('CUDA_VISIBLE_DEVICES=' + str(arguments['gpus']) + ' python3 main_1gpu.py ' + args_str)

os.system('CUDA_VISIBLE_DEVICES=' + str(arguments['gpus']) + ' python3 main_1gpu.py ' + args_str )
