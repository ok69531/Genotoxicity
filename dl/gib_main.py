import logging
import argparse
import warnings

import numpy as np
from copy import deepcopy

import torch
from torch.optim import Adam
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from module.load_dataset import GenoDataset
from module.utils import set_seed

# GIB
from arguments.gib_args import gib_args
from gib_model.gib import (
    GIBGIN,
    Discriminator,
    gib_train,
    gib_eval
)

warnings.filterwarnings('ignore')
logging.basicConfig(format = '', level = logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument('--model', type = str, default = 'gib', help = 'gib, vgib, pgib, gsat')
parser.add_argument('--tg_num', type = int, default = 471, help = 'OECD TG for Genotoxicity (471, 473, 476, 487 / 474, 475, 478, 483, 486, 488)')
parser.add_argument('--target', type = str, default = 'maj', help = 'maj or consv')
parser.add_argument('--train_frac', type = float, default = 0.7)
parser.add_argument('--val_frac', type = float, default = 0.1)
try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])


# def main():
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Cuda Available: {torch.cuda.is_available()}, {device}')

dataset = GenoDataset(root = 'dataset', tg_num = 471)

avg_nodes = 0.0
avg_edge_index = 0.0
for i in range(len(dataset)):
    avg_nodes += dataset[i].x.shape[0]
    avg_edge_index += dataset[i].edge_index.shape[1]

avg_nodes /= len(dataset)
avg_edge_index /= len(dataset)
logging.info('graphs {}, avg_nodes {:.4f}, avg_edge_index {:.4f}'.format(len(dataset), avg_nodes, avg_edge_index/2))

val_losses, val_aucs, val_f1s, val_accs, val_precs, val_recs = [], [], [], [], [], []
test_losses, test_aucs, test_f1s, test_accs, test_precs, test_recs = [], [], [], [], [], []
params_list = []
optim_params_list = []









logging.info('')
logging.info('Model: {}'.format(args.model))
logging.info('TG: {}'.format(args.tg_num))
logging.info('Target: {}'.format(args.target))

logging.info('')
logging.info('test f1: ${{{:.3f}}}_{{\\pm {:.3f}}}$'.format(np.mean(test_f1s) * 100, np.std(test_f1s) * 100))
logging.info('test precision: ${{{:.3f}}}_{{\\pm {:.3f}}}$'.format(np.mean(test_precs) * 100, np.std(test_precs) * 100))
logging.info('test recall: ${{{:.3f}}}_{{\\pm {:.3f}}}$'.format(np.mean(test_recs) * 100, np.std(test_recs) * 100))
logging.info('test accuracy: ${{{:.3f}}}_{{\\pm {:.3f}}}$'.format(np.mean(test_accs) * 100, np.std(test_accs) * 100))
logging.info('test roc-auc: ${{{:.3f}}}_{{\\pm {:.3f}}}$'.format(np.mean(test_aucs) * 100, np.std(test_aucs) * 100))
        
