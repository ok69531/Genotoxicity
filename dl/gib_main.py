import os
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

from arguments.data_args import (
    tg471_args,
    tg473_args,
    tg474_args,
    tg475_args,
    tg476_args,
    tg478_args,
    tg487_args
)

# gib
from arguments.model_args.gib_args import gib_args
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
parser.add_argument('--train_frac', type = float, default = 0.8)
parser.add_argument('--val_frac', type = float, default = 0.1)
try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Cuda Available: {torch.cuda.is_available()}, {device}')
    
    if args.tg_num == 471: tg_args = tg471_args
    elif args.tg_num == 473: tg_args = tg473_args
    elif args.tg_num == 474: tg_args = tg474_args
    elif args.tg_num == 475: tg_args = tg475_args
    elif args.tg_num == 476: tg_args = tg476_args
    elif args.tg_num == 478: tg_args = tg478_args
    elif args.tg_num == 487: tg_args = tg487_args
    else: raise ValueError(f'TG {args.tg_num} not supported.')
    
    args.hidden_dim = tg_args.hidden
    args.num_layers = tg_args.num_layers
    args.batch_size = tg_args.btach_size
    args.epochs = tg_args.epochs
    args.optimizer = tg_args.optimizer
    args.lr = tg_args.lr
    args.weight_decay = tg_args.weight_decay
    
    if args.tg_num == 471:
        gib_args.inner_loop = 70; gib_args.beta = 0.3; gib_args.pp_weight = 0.5
    elif args.tg_num == 473:
        gib_args.inner_loop = 70; gib_args.beta = 0.1; gib_args.pp_weight = 0.1
    elif args.tg_num == 476:
        gib_args.inner_loop = 30; gib_args.beta = 0.1; gib_args.pp_weight = 0.9
    elif args.tg_num == 474:
        gib_args.inner_loop = 50; gib_args.beta = 0.7; gib_args.pp_weight = 0.9
    
    logging.info(args)
    
    dataset = GenoDataset(root = 'dataset', tg_num = args.tg_num)
    for i in range(len(dataset)):
        if dataset[i].x.shape[0] == 1:
            print(i, dataset[i].x.shape)

    if args.tg_num == 471:
        remove_idx = [1616, 2896]
    elif args.tg_num == 473:
        remove_idx = [422, 1121, 1463, 1871, 1987, 2076]
    elif args.tg_num == 476:
        remove_idx = [429, 1111, 1491, 1535, 1802, 2028]
    elif args.tg_num == 474:
        remove_idx = [662, 1073, 1146, 1277]
    elif args.tg_num == 475:
        remove_idx = [100]
    else:
        remove_idx = []

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
    seed = seeds[0]
    for seed in seeds:
        logging.info(f'======================= Run: {seeds.index(seed)} =================')
        set_seed(seed)

        num_train = int(len(dataset) * args.train_frac)
        num_valid = int(len(dataset) * args.val_frac)
        num_test = len(dataset) - (num_train + num_valid)
        assert num_train + num_valid + num_test == len(dataset)

        indices = torch.arange(len(dataset))
        train_idx, val_idx, test_idx = random_split(indices, [num_train, num_valid, num_test])

        if (args.model == 'gib') or (args.model == 'vgib'):
            train_idx = [x.item() for x in train_idx if x.item() not in remove_idx]
            val_idx = [x.item() for x in val_idx if x.item() not in remove_idx]
            test_idx = [x.item() for x in test_idx if x.item() not in remove_idx]

            train_loader = DataLoader(dataset[train_idx], batch_size = args.batch_size, shuffle = True)
            val_loader = DataLoader(dataset[val_idx], batch_size = args.batch_size, shuffle = False)
            test_loader = DataLoader(dataset[test_idx], batch_size = args.batch_size, shuffle = False)
        else:
            train_loader = DataLoader(dataset[list(train_idx)], batch_size = args.batch_size, shuffle = True)
            val_loader = DataLoader(dataset[list(val_idx)], batch_size = args.batch_size, shuffle = False)
            test_loader = DataLoader(dataset[list(test_idx)], batch_size = args.batch_size, shuffle = False)

        if args.model == 'gib':
            model = GIBGIN(dataset.num_classes, args.num_layers, args.hidden).to(device)
            discriminator = Discriminator(args.hidden_dim).to(device)
            optimizer = Adam(model.parameters(), lr = args.lr)
            optimizer_local = Adam(discriminator.parameters(), lr = args.lr)

        best_val_loss, best_val_auc, best_val_f1 = 100, 0, 0
        final_test_loss, final_test_auc, final_test_f1 = 100, 0, 0

        for epoch in range(1, args.epochs + 1):
            if args.model == 'gib':
                train_loss = gib_train(model, discriminator, optimizer, optimizer_local, device, train_loader, gib_args, args)
                val_loss, val_sub_metrics, _ = gib_eval(model, device, val_loader, args)
                test_loss, test_sub_metrics, _ = gib_eval(model, device, test_loader, args)

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
                
                if args.model == 'gib':
                    params = (deepcopy(model.state_dict(), deepcopy(discriminator.state_dict())))
                    optim_params = (deepcopy(optimizer.state_dict(), deepcopy(optimizer_local.state_dict())))
                
        val_losses.append(best_val_loss); test_losses.append(final_test_loss)
        val_aucs.append(best_val_auc); test_aucs.append(final_test_auc)
        val_f1s.append(best_val_f1); test_f1s.append(final_test_f1)
        val_accs.append(best_val_acc); test_accs.append(final_test_acc)
        val_precs.append(best_val_prec); test_precs.append(final_test_prec)
        val_recs.append(best_val_rec); test_recs.append(final_test_rec)
        params_list.append(params); optim_params_list.append(optim_params)
    
    checkpoints = {
        'params_dict': params_list,
        'optim_dict': optim_params_list,
        'val_f1s': val_f1s,
        'test_f1s': test_f1s
    }
    
    save_path = f'saved_result/tg{args.tg_num}'
    if os.path.isdir(save_path):
        pass
    else:
        os.makedirs(save_path)
    save_path = os.path.join(save_path, f'{args.target}_tg{args.tg_num}_{args.model}.pt')
    torch.save(checkpoints, save_path)
    
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


if __name__ == '__main__':
    main()


#%%
#     for seed in range(args.num_runs):
#         if args.model == 'gib':
#             model = gibGIN(dataset.num_classes, gib_args.num_layers, gib_args.hidden).to(device)
#             discriminator = Discriminator(gib_args.hidden).to(device)
#             optimizer = Adam(model.parameters(), lr = gib_args.lr)
#             optimizer_local = Adam(discriminator.parameters(), lr = gib_args.lr)
#             max_epochs = gib_args.epochs
            
        
#         elif args.model == 'GIN':
#             model = GraphIsomorphismNetwork(dataset.num_classes, args).to(device)
#             criterion = torch.nn.CrossEntropyLoss()
#             optimizer = Adam(model.parameters(), lr = args.lr)
#             max_epochs = 100
        
#         elif args.model == 'GCN':
#             model = GraphConvolutionalNetwork(dataset.num_classes, args).to(device)
#             criterion = torch.nn.CrossEntropyLoss()
#             optimizer = Adam(model.parameters(), lr = args.lr)
#             max_epochs = 100
        
#         best_val_loss, best_val_auc, best_val_f1 = 0, 0, 0
#         final_test_loss, final_test_auc, final_test_f1 = 0, 0, 0
        
#         for epoch in range(1, max_epochs+1):
#             if args.model == 'gib':
#                 train_loss = gib_train(model, discriminator, optimizer, optimizer_local, device, train_loader, gib_args)
#                 val_loss, val_sub_metrics, _ = gib_eval(model, device, val_loader)
#                 test_loss, test_sub_metrics, _ = gib_eval(model, device, test_loader)
                
            
#             elif args.model == 'GIN':
#                 train_loss, _ = gin_train(model, optimizer, device, train_loader, criterion)
#                 val_loss, val_sub_metrics, _ = gin_evaluation(model, device, val_loader, criterion)
#                 test_loss, test_sub_metrics, _ = gin_evaluation(model, device, test_loader, criterion)
            
#             elif args.model == 'GCN':
#                 train_loss, _ = gcn_train(model, optimizer, device, train_loader, criterion)
#                 val_loss, val_sub_metrics, _ = gcn_evaluation(model, device, val_loader, criterion)
#                 test_loss, test_sub_metrics, _ = gcn_evaluation(model, device, test_loader, criterion)
            
#             logging.info('=== epoch: {}'.format(epoch))
#             logging.info('Train loss: {:.5f} | Validation loss: {:.5f}, Auc: {:.5f}, F1: {:.5f} | Test loss: {:.5f}, Auc: {:.5f}, F1: {:.5f}'.format(
#                              train_loss, val_loss, val_sub_metrics['auc'], val_sub_metrics['f1'],
#                              test_loss, test_sub_metrics['auc'], test_sub_metrics['f1']))
            
#             if (val_sub_metrics['f1'] > best_val_f1) or \
#                 ((val_loss < best_val_loss) and (val_sub_metrics['f1'] == best_val_f1)):
#                 best_val_loss = val_loss
#                 best_val_f1 = val_sub_metrics['f1']; best_val_auc = val_sub_metrics['auc']
#                 best_val_acc = val_sub_metrics['accuracy']; best_val_prec = val_sub_metrics['precision']; best_val_rec = val_sub_metrics['recall']
#                 final_test_loss = test_loss
#                 final_test_f1 = test_sub_metrics['f1']; final_test_auc = test_sub_metrics['auc']
#                 final_test_acc = test_sub_metrics['accuracy']; final_test_prec = test_sub_metrics['precision']; final_test_rec = test_sub_metrics['recall']
                
#                 if args.model == 'gib':
#                     params = (deepcopy(model.state_dict(), deepcopy(discriminator.state_dict())))
#                     optim_params = (deepcopy(optimizer.state_dict(), deepcopy(optimizer_local.state_dict())))
#                 elif args.model == 'Vgib':
#                     params = (deepcopy(model.state_dict(), deepcopy(classifier.state_dict())))
#                     optim_params = deepcopy(optimizer.state_dict())
#                 else:
#                     params = deepcopy(model.state_dict())
#                     optim_params = deepcopy(optimizer.state_dict())
                
#         val_losses.append(best_val_loss); test_losses.append(final_test_loss)
#         val_aucs.append(best_val_auc); test_aucs.append(final_test_auc)
#         val_f1s.append(best_val_f1); test_f1s.append(final_test_f1)
#         val_accs.append(best_val_acc); test_accs.append(final_test_acc)
#         val_precs.append(best_val_prec); test_precs.append(final_test_prec)
#         val_recs.append(best_val_rec); test_recs.append(final_test_rec)
#         params_list.append(params); optim_params_list.append(optim_params)
    
#     checkpoints = {
#         'params_dict': params_list,
#         'optim_dict': optim_params_list,
#         'val_accs': val_aucs,
#         'test_accs': test_aucs,
#         'val_f1s': val_f1s,
#         'test_f1s': test_f1s
#     }
    
#     path = f'saved_model/{args.model}/{args.dataset}'
#     file_name = f'{args.model}_{args.dataset}'
#     save_model(checkpoints, path, file_name)

# if __name__ == '__main__':
#     main()