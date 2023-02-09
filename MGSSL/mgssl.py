#%%
from importlib.util import module_from_spec
import os
import random
import shutil
import pickle
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

import utils
from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, classification_report

from itertools import compress, repeat, product, chain
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold


#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.data import Data, DataLoader, InMemoryDataset

print('Version of pytorch: ', torch.__version__)
print('Is GPU available?: ', torch.cuda.is_available())
print('GPU: ', torch.cuda.get_device_name(0))


#%%
device = torch.device("cuda:0")

dataset_name = "bace"

if dataset_name == "tox21":
    num_task = 12
elif dataset_name == "hiv":
    num_task = 1
elif dataset_name == "pcba":
    num_task = 128
elif dataset_name == "muv":
    num_task = 17
elif dataset_name == "bace":
    num_task = 1
elif dataset_name == "bbbp":
    num_task = 1
elif dataset_name == "toxcast":
    num_task = 617
elif dataset_name == "sider":
    num_task = 27
elif dataset_name == "clintox":
    num_task = 2
elif dataset_name == 'geno':
    num_task = 1



#%%
def train(model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()

        optimizer.step()

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def eval(model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()
    
    y_pred = np.where(sigmoid(y_scores) > 0.5, 1.0, 0.0)
    
    #acc_list = []
    roc_list = []
    
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))
            #acc_list.append(np.mean((y_true[is_valid, i] +1)/2 == y_pred[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list), roc_list, y_pred 


#%%
dataset = MoleculeDataset('dataset/' + dataset_name , dataset = dataset_name)
smiles_list = pd.read_csv('dataset/' + dataset_name + '/processed/smiles.csv', header=None)[0].tolist()

num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 


#%%
tast_roc_list = []
test_prec_list = []
test_recall_list = []
test_f1_list = []
test_acc_list = []

for seed in range(50):

    train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed=seed)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers = 0)
    val_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers = 0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers = 0)

    # from tensorboardX import SummaryWriter

    model = GNN_graphpred(5, 300, num_tasks=num_task, JK='last', drop_ratio=0.2, graph_pooling='mean', gnn_type='gin').to(device)

    model.from_pretrained('saved_model/init.pth')

    model.to(device)

    criterion = nn.BCEWithLogitsLoss(reduction = "none")

    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})

    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":0.001*1})
    optimizer = optim.Adam(model_param_group, lr=0.001, weight_decay=0)
    print(optimizer)

    eval_train = 1

    for epoch in range(1, 15+1):
        # print("====epoch " + str(epoch))
        
        train(model, device, train_loader, optimizer)

        # print("====Evaluation")
        if eval_train:
            train_acc, train_task_acc, train_pred = eval(model, device, train_loader)
        else:
            # print("omit the training accuracy computation")
            train_acc = 0
        val_acc, val_task_acc, val_pred = eval(model, device, val_loader)
        test_acc, test_task_acc, test_pred = eval(model, device, test_loader)

        print("train: %f val: %f test: %f" %(train_acc, val_acc, test_acc))

    print("Seed:", seed)
    
    test_y = []
    for d, s in enumerate(test_dataset):
        y_tmp = [0 if i == -1 else i for i in s.y.numpy()]
        test_y.append(y_tmp[0])
        
    pred = [int(i[0]) for i in test_pred]
    
    tast_roc_list.append(test_task_acc)
    test_acc_list.append(accuracy_score(test_y, pred))
    test_prec_list.append(precision_score(test_y, pred))
    test_recall_list.append(recall_score(test_y, pred))
    test_f1_list.append(f1_score(test_y, pred))


    
#%%
# column_average = [sum(sub_list) / len(sub_list) for sub_list in zip(*tast_roc_list)]
row_average = [sum(sub_list) / len(sub_list) for sub_list in tast_roc_list]


from scipy.stats import sem

# print(np.mean(row_average))
# print(sem(row_average))
# print(np.std(row_average))

print('Precision: ', np.mean(test_prec_list).round(3), '(', sem(test_prec_list).round(3), ')',
      '\nRecall: ', np.mean(test_recall_list).round(3), '(', sem(test_recall_list).round(3), ')',
      '\nF1: ', np.mean(test_f1_list).round(3), '(', sem(test_f1_list).round(3), ')',
      '\nAUC: ', np.mean(row_average).round(3), '(', sem(row_average).round(3), ')',
      '\nAccuracy: ', np.mean(test_acc_list).round(3), '(', sem(test_acc_list).round(3), ')'
      )


#%%
test_y = []
for d, s in enumerate(test_dataset):
    y_tmp = [0 if i == -1 else i for i in s.y.numpy()]
    test_y.append(y_tmp[0])


pred = [int(i[0]) for i in test_pred]

pd.crosstab(test_y, pred, rownames = ['true'], colnames = ['pred'])
print(classification_report(test_y, pred, digits = 3))




