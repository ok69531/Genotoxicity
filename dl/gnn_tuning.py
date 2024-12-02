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
from module.utils import set_seed, get_seed

import wandb

# gib
from gib_model.gin import (
    GraphIsomorphismNetwork,
    gin_train,
    gin_evaluation
)

warnings.filterwarnings('ignore')
logging.basicConfig(format = '', level = logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument('--model', type = str, default = 'gin')
parser.add_argument('--tg_num', type = int, default = 471, help = 'OECD TG for Genotoxicity (471, 473, 476, 487 / 474, 475, 478, 483, 486, 488)')
parser.add_argument('--target', type = str, default = 'maj', help = 'maj or consv')
parser.add_argument('--train_frac', type = float, default = 0.8)
parser.add_argument('--val_frac', type = float, default = 0.1)
parser.add_argument('--batch_size', type = int, default = 128)
parser.add_argument('--readout', type = str, default = 'max')
parser.add_argument('--hidden_dim', type = int, default = 128)
parser.add_argument('--num_layers', type = int, default = 5)
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--epochs', type = int, default = 100)
try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])


wandb.login(key = open('wandb_key.txt', 'r').readline())
sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'avg val f1'},
    'parameters':{
        'batch_size': {'values': [32, 64, 128]},
        'hidden_dim': {'values': [32, 64, 128, 300, 512]},
        'num_layers': {'values': [2, 3, 4, 5, 6, 7]},
        'lr': {'values': [0.001, 0.003]},
        'epochs': {'values': [100, 300]}
    }       
}
sweep_id = wandb.sweep(sweep_configuration, project = f'gnn_tg{args.tg_num}_genotoxicity')


def main():
    wandb.init()
    
    args.batch_size = wandb.config.batch_size
    args.hidden_dim = wandb.config.hidden_dim
    args.num_layers = wandb.config.num_layers
    args.lr = wandb.config.lr
    args.epochs = wandb.config.epochs
    
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

    seeds = get_seed(args.tg_num)
    seed=seeds[0]
    # for seed in seeds:
    logging.info(f'======================= Run: {seeds.index(seed)} =================')
    set_seed(seed)

    num_train = int(len(dataset) * args.train_frac)
    num_valid = int(len(dataset) * args.val_frac)
    num_test = len(dataset) - (num_train + num_valid)
    assert num_train + num_valid + num_test == len(dataset)

    indices = torch.arange(len(dataset))
    train_idx, val_idx, test_idx = random_split(indices, [num_train, num_valid, num_test])

    train_loader = DataLoader(dataset[list(train_idx)], batch_size = args.batch_size, shuffle = True)
    val_loader = DataLoader(dataset[list(val_idx)], batch_size = args.batch_size, shuffle = False)
    test_loader = DataLoader(dataset[list(test_idx)], batch_size = args.batch_size, shuffle = False)


    if args.model == 'gin':
        model = GraphIsomorphismNetwork(dataset.num_classes, args).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 10.]).to(device))
        optimizer = Adam(model.parameters(), lr = args.lr)

    best_val_loss, best_val_auc, best_val_f1 = 100, 0, 0
    final_test_loss, final_test_auc, final_test_f1 = 100, 0, 0

    for epoch in range(1, args.epochs+1):
        if args.model == 'gin':
            train_loss, _ = gin_train(model, optimizer, device, train_loader, criterion, args)
            val_loss, val_sub_metrics, _ = gin_evaluation(model, device, val_loader, criterion, args)
            test_loss, test_sub_metrics, _ = gin_evaluation(model, device, test_loader, criterion, args)

        logging.info('=== epoch: {}'.format(epoch))
        logging.info('Train loss: {:.5f} | Validation loss: {:.5f}, Auc: {:.5f}, F1: {:.5f} | Test loss: {:.5f}, Auc: {:.5f}, F1: {:.5f}'.format(
                            train_loss, val_loss, val_sub_metrics['auc'], val_sub_metrics['f1'],
                            test_loss, test_sub_metrics['auc'], test_sub_metrics['f1']))

        if (val_sub_metrics['f1'] > best_val_f1) or \
            ((val_loss < best_val_loss) and (val_sub_metrics['f1'] == best_val_f1)):
            best_val_loss = val_loss
            best_val_f1 = val_sub_metrics['f1']; best_val_auc = val_sub_metrics['auc']
            best_val_acc = val_sub_metrics['accuracy']; best_val_prec = val_sub_metrics['precision']; best_val_rec = val_sub_metrics['recall']
            final_test_loss = test_loss
            final_test_f1 = test_sub_metrics['f1']; final_test_auc = test_sub_metrics['auc']
            final_test_acc = test_sub_metrics['accuracy']; final_test_prec = test_sub_metrics['precision']; final_test_rec = test_sub_metrics['recall']
            
            if args.model == 'gin':
                params = deepcopy(model.state_dict())
                optim_params = deepcopy(optimizer.state_dict())
            
    val_losses.append(best_val_loss); test_losses.append(final_test_loss)
    val_aucs.append(best_val_auc); test_aucs.append(final_test_auc)
    val_f1s.append(best_val_f1); test_f1s.append(final_test_f1)
    val_accs.append(best_val_acc); test_accs.append(final_test_acc)
    val_precs.append(best_val_prec); test_precs.append(final_test_prec)
    val_recs.append(best_val_rec); test_recs.append(final_test_rec)
    params_list.append(params); optim_params_list.append(optim_params)

    wandb.log({
        'avg val f1': np.mean(val_f1s),
        'avg test f1': np.mean(test_f1s)
    })

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

wandb.agent(sweep_id = sweep_id, function = main, count = 300)
