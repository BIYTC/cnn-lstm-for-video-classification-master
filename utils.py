import argparse
import json
import os
import sys
from pprint import pprint

import numpy as np
from easydict import EasyDict as edict
from PIL import Image
from shutil import copyfile
import torch


def parse_args():
    """
    Parse the arguments of the program
    :return: (config_args)
    :rtype: tuple
    """
    # Create a parser
    parser = argparse.ArgumentParser(description="CNN-LSTM PyTorch Implementation")
    parser.add_argument('--version', action='version', version='%(prog)s 0.0.1')
    parser.add_argument('--config', default='TrainConfig.json', type=str, help='Configuration file')

    # Parse the arguments
    args = parser.parse_args()

    # Parse the configurations from the config json file provided
    try:
        if args.config is not None:
            with open(args.config, 'r') as config_file:
                config_args_dict = json.load(config_file)
        else:
            print("Add a config file using \'--config file_name.json\'", file=sys.stderr)
            exit(1)

    except FileNotFoundError:
        print("ERROR: Config file not found: {}".format(args.config), file=sys.stderr)
        exit(1)
    except json.decoder.JSONDecodeError:
        print("ERROR: Config file is not a proper JSON file!", file=sys.stderr)
        exit(1)

    config_args = edict(config_args_dict)

    pprint(config_args)
    print("\n")

    return config_args


def create_experiment_dirs(exp_dir):
    """
    Create Directories of a regular tensorflow experiment directory
    :param exp_dir:
    :return summary_dir, checkpoint_dir:
    """
    experiment_dir = os.path.realpath(
        os.path.join(os.path.dirname(__file__))) + "/experiments/" + exp_dir + "/"
    summary_dir = experiment_dir + 'summaries/'
    checkpoint_dir = experiment_dir + 'checkpoints/'

    dirs = [summary_dir, checkpoint_dir]
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        copyfile('./TrainConfig.json', experiment_dir + 'TrainConfig.json')

        print("Experiment directories created!")
        # return experiment_dir, summary_dir, checkpoint_dir
        return experiment_dir, summary_dir, checkpoint_dir
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


# def calc_dataset_stats(dataset, axis=0, ep=1e-7):
#     BatchNum = 2000
#     mean = np.zeros(1)
#     std = np.zeros(1)
#     Nt = dataset.shape[0] // BatchNum
#     if Nt > 0:
#         for i in range(Nt):
#             mean += np.mean(dataset[i * BatchNum:(i + 1) * BatchNum], axis=axis) / 255.0 * BatchNum
#             std += (np.std(dataset[i * BatchNum:(i + 1) * BatchNum] + ep, axis=axis) / 255.0) ** 2 * (BatchNum)
#
#         mean += np.mean(dataset[Nt * BatchNum:dataset.shape[0]], axis=axis) / 255.0 * (dataset.shape[0] - Nt * BatchNum)
#         std += (np.std(dataset[Nt * BatchNum:dataset.shape[0]] + ep, axis=axis) / 255.0) ** 2 * (
#             dataset.shape[0] - Nt * BatchNum)
#         mean = mean / dataset.shape[0]
#         std = (std / (dataset.shape[0])) ** 0.5
#         # pass
#     else:
#         mean = np.mean(dataset, axis=axis) / 255.0
#         std = np.std(dataset + ep, axis=axis) / 255.0
#         # pass
#
#     return mean.tolist(), std.tolist()


class AverageTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



# def load_mean_std(filename):
#     # filename = self.args.checkpoint_dir + filename
#     try:
#         print("Loading mean and std '{}'".format(filename))
#         checkpoint = torch.load(filename)
#         mean = checkpoint['mean']
#         std = checkpoint['std']
#         print("Load mean and std successfully from '{}' at (epoch {})\n".
#               format(filename, checkpoint['epoch']))
#         return mean, std
#     except:
#         print("No mean and std exists from '{}'. Skipping...\n".format(filename))


# def catlog(label):
#     log = [770, 790, 810, 830, 850, 870, 890, 910, 930, 950, 970, 1000, 1040, ]  # 除两侧区间区间外，均为左开右闭
#     if label < 750:
#         return 0
#     for idx, i in enumerate(log, 1):
#         if i >= label:
#             return idx
#     return 14
def catlog(label):
    log = [900, 1000, ]  # 除两侧区间区间外，均为左开右闭
    if label < 800:
        return 0
    for idx, i in enumerate(log, 1):
        if i >= label:
            return idx
    return 3


import os


# loading partition and labels
def load_data(dir_path):
    labels = {}
    partition = {}

    train_list = []
    data = open(os.path.join(dir_path, 'train.txt'), 'r').read()
    c = data.split('\n')
    for i in c[:len(c) - 1]:
        labels.update({i.split(' ')[0]: int(i.split(' ')[1])})
        train_list.append(i.split(' ')[0])

    val_list = []
    data = open(os.path.join(dir_path, 'val.txt'), 'r').read()
    c = data.split('\n')
    for i in c[:len(c) - 1]:
        labels.update({i.split(' ')[0]: int(i.split(' ')[1])})
        val_list.append(i.split(' ')[0])

    partition['train'] = train_list
    partition['val'] = val_list
    return partition, labels


# converting labels to integers
def convert_to_integer(path):
    dict_labels = {}
    a = open(path, 'r').read()
    c = a.split('\n')
    for i in c[:len(c) - 1]:
        dict_labels.update({i.split(' ')[1]: i.split(' ')[0]})
    return dict_labels


if __name__ == "__main__":
    a = catlog(1001)
    print(a)
