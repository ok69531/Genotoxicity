#%%
from importlib.util import module_from_spec
import os
import random
import pickle
import openpyxl

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from tqdm import tqdm

from utils import *
from model import GNN, GNN_extract
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, classification_report

from itertools import compress, repeat, product, chain
from sklearn.model_selection import StratifiedKFold


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

dataset_name = "geno"

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
dataset = MoleculeDataset('dataset/' + dataset_name , dataset = dataset_name)
smiles_list = pd.read_csv('dataset/' + dataset_name + '/processed/smiles.csv', header=None)[0].tolist()

num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 


#%%
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

loader = DataLoader(dataset, batch_size = 32, shuffle = False, num_workers = 0)

model = GNN_extract(5, 300, num_tasks=num_task, JK='last', drop_ratio=0.2, graph_pooling='mean', gnn_type='gin').to(device)
model.from_pretrained('saved_model/init.pth')
model.eval()
model.to(device)

feature = []
y_true = []
for step, batch in enumerate(loader):
    batch = batch.to(device)
    
    with torch.no_grad():
        feature_ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

    feature.append(feature_)
    y_true.append(batch.y.view(len(feature_)))

feature = torch.cat(feature, dim = 0).cpu().numpy()
y_true = torch.cat(y_true, dim = 0).cpu().numpy()

np.unique(y_true)

y_true = [0 if i == -1 else 1 for i in y_true]


print('count', '\n', pd.Series(y_true).value_counts(),
      '\nratio', '\n',  pd.Series(y_true).value_counts(normalize = True).round(3))



#%%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

logit_auc = []
logit_prec = []
logit_recall = []
logit_acc = []
logit_f1 = []

for i in tqdm(range(50)):
    x_train, x_test, y_train, y_test = train_test_split(feature, y_true, test_size = 0.20, random_state = i)
    
    sm = SMOTE(random_state = i)
    x_train, y_train = sm.fit_sample(x_train, y_train)
        
    logit = LogisticRegression(random_state = i)
    logit.fit(x_train, y_train)
    
    # logit_pred = logit.predict(x_test)
    logit_pred_prob = logit.predict_proba(x_test)[:, 1]
    logit_pred = [1 if i > 0.5 else 0 for i in logit_pred_prob]
    
    logit_auc.append(roc_auc_score(y_test, logit_pred_prob))
    logit_prec.append(precision_score(y_test, logit_pred))
    logit_recall.append(recall_score(y_test, logit_pred))
    logit_acc.append(accuracy_score(y_test, logit_pred))
    logit_f1.append(f1_score(y_test, logit_pred))


print('auc: ', np.mean(logit_auc).round(3), '(', (np.std(logit_auc, ddof = 1) / np.sqrt(len(logit_auc))).round(3), ')',
      '\nprecision: ', np.mean(logit_prec).round(3), '(', (np.std(logit_prec, ddof = 1)/np.sqrt(len(logit_prec))).round(3), ')',
      '\nrecall: ', np.mean(logit_recall).round(3), '(', (np.std(logit_recall, ddof = 1)/np.sqrt(len(logit_recall))).round(3), ')',
      '\nacuracy: ', np.mean(logit_acc).round(3), '(', (np.std(logit_acc, ddof = 1)/np.sqrt(len(logit_acc))).round(3) ,')',
      '\nf1: ', np.mean(logit_f1).round(3), '(', (np.std(logit_f1, ddof = 1)/np.sqrt(len(logit_f1))).round(3), ')')


#%%
from sklearn.svm import SVC

svm_auc = []
svm_prec = []
svm_recall = []
svm_acc = []
svm_f1 = []

for i in tqdm(range(50)):
    x_train, x_test, y_train, y_test = train_test_split(feature, y_true, test_size = 0.2, random_state = i)
    
    sm = SMOTE(random_state = i)
    x_train, y_train = sm.fit_sample(x_train, y_train)
    
    svm = SVC(random_state = i, probability = True)
    svm.fit(x_train, y_train)
    
    # svm_pred = svm.predict(x_test)
    svm_pred_prob = svm.predict_proba(x_test)[:, 1]
    svm_pred = [1 if i > 0.5 else 0 for i in svm_pred_prob]
    
    svm_auc.append(roc_auc_score(y_test, svm_pred_prob))
    svm_prec.append(precision_score(y_test, svm_pred))
    svm_recall.append(recall_score(y_test, svm_pred))
    svm_acc.append(accuracy_score(y_test, svm_pred))
    svm_f1.append(f1_score(y_test, svm_pred))


print('auc: ', np.mean(svm_auc).round(3), '(', (np.std(svm_auc, ddof = 1) / np.sqrt(len(svm_auc))).round(3), ')',
      '\nprecision: ', np.mean(svm_prec).round(3), '(', (np.std(svm_prec, ddof = 1)/np.sqrt(len(svm_prec))).round(3), ')',
      '\nrecall: ', np.mean(svm_recall).round(3), '(', (np.std(svm_recall, ddof = 1)/np.sqrt(len(svm_recall))).round(3), ')',
      '\nacuracy: ', np.mean(svm_acc).round(3), '(', (np.std(svm_acc, ddof = 1)/np.sqrt(len(svm_acc))).round(3) ,')',
      '\nf1: ', np.mean(svm_f1).round(3), '(', (np.std(svm_f1, ddof = 1)/np.sqrt(len(svm_f1))).round(3), ')')


#%%
from sklearn.ensemble import RandomForestClassifier

rf_auc = []
rf_prec = []
rf_recall = []
rf_acc = []
rf_f1 = []

for i in tqdm(range(50)):
    x_train, x_test, y_train, y_test = train_test_split(feature, y_true, test_size = 0.20, random_state = i)
    
    sm = SMOTE(random_state = i)
    x_train, y_train = sm.fit_sample(x_train, y_train)
    
    rf = RandomForestClassifier(random_state = i)
    rf.fit(x_train, y_train)
    
    # rf_pred = rf.predict(x_test)
    rf_pred_prob = rf.predict_proba(x_test)[:, 1]
    rf_pred = [1 if i > 0.5 else 0 for i in rf_pred_prob]
    
    rf_auc.append(roc_auc_score(y_test, rf_pred_prob))
    rf_prec.append(precision_score(y_test, rf_pred))
    rf_recall.append(recall_score(y_test, rf_pred))
    rf_acc.append(accuracy_score(y_test, rf_pred))
    rf_f1.append(f1_score(y_test, rf_pred))


print('auc: ', np.mean(rf_auc).round(3), '(', (np.std(rf_auc, ddof = 1) / np.sqrt(len(rf_auc))).round(3), ')',
      '\nprecision: ', np.mean(rf_prec).round(3), '(', (np.std(rf_prec, ddof = 1)/np.sqrt(len(rf_prec))).round(3), ')',
      '\nrecall: ', np.mean(rf_recall).round(3), '(', (np.std(rf_recall, ddof = 1)/np.sqrt(len(rf_recall))).round(3), ')',
      '\nacuracy: ', np.mean(rf_acc).round(3), '(', (np.std(rf_acc, ddof = 1)/np.sqrt(len(rf_acc))).round(3) ,')',
      '\nf1: ', np.mean(rf_f1).round(3), '(', (np.std(rf_f1, ddof = 1)/np.sqrt(len(rf_f1))).round(3), ')')


#%%
from itertools import product
from collections.abc import Iterable
from sklearn.model_selection import StratifiedKFold

def ParameterGrid(param_dict):
    if not isinstance(param_dict, dict):
        raise TypeError('Parameter grid is not a dict ({!r})'.format(param_dict))
    
    if isinstance(param_dict, dict):
        for key in param_dict:
            if not isinstance(param_dict[key], Iterable):
                raise TypeError('Parameter grid value is not iterable '
                                '(key={!r}, value={!r})'.format(key, param_dict[key]))
    
    items = sorted(param_dict.items())
    keys, values = zip(*items)
    
    params_grid = []
    for v in product(*values):
        params_grid.append(dict(zip(keys, v))) 
    
    return params_grid


def BinaryCV(x, y, model, params_grid):
    skf = StratifiedKFold(n_splits = 5)
    
    result_ = []
    metrics = ['macro_precision', 'macro_recall', 'macro_f1', 
               'accuracy', 'tau', 'auc']
    
    train_metrics = list(map(lambda x: 'train_' + x, metrics))
    val_metrics = list(map(lambda x: 'val_' + x, metrics))
    
    for i in tqdm(range(len(params_grid))):
        train_macro_precision_ = []
        train_macro_recall_ = []
        train_macro_f1_ = []
        train_accuracy_  = []
        train_auc_ = []
        
        val_macro_precision_ = []
        val_macro_recall_ = []
        val_macro_f1_ = []
        val_accuracy_ = []
        val_auc_ = []
        
        for train_idx, val_idx in skf.split(x, y):
            train_x, train_y = x.iloc[train_idx], y.iloc[train_idx]
            val_x, val_y = x.iloc[val_idx], y.iloc[val_idx]
            
            clf = model(**params_grid[i])
            clf.fit(np.array(train_x))
            
            train_pred_prob = abs(clf.score_samples(np.array(train_x)))
            val_pred_prob = abs(clf.score_samples(np.array(val_x)))
            
            train_pred = [1 if i-0.5>0 else 0 for i in train_pred_prob]
            val_pred = [1 if i-0.5>0 else 0 for i in val_pred_prob]
            
            train_macro_precision_.append(precision_score(train_y, train_pred))
            train_macro_recall_.append(recall_score(train_y, train_pred))
            train_macro_f1_.append(f1_score(train_y, train_pred))
            train_accuracy_.append(accuracy_score(train_y, train_pred))
            train_auc_.append(roc_auc_score(train_y, train_pred_prob))

            val_macro_precision_.append(precision_score(val_y, val_pred))
            val_macro_recall_.append(recall_score(val_y, val_pred))
            val_macro_f1_.append(f1_score(val_y, val_pred))
            val_accuracy_.append(accuracy_score(val_y, val_pred))
            val_auc_.append(roc_auc_score(val_y, val_pred_prob))
            
        result_.append(dict(
            zip(list(params_grid[i].keys()) + train_metrics + val_metrics, 
                list(params_grid[i].values()) + 
                [np.mean(train_macro_precision_), 
                 np.mean(train_macro_recall_), 
                 np.mean(train_macro_f1_), 
                 np.mean(train_accuracy_), 
                 np.mean(train_auc_),
                 np.mean(val_macro_precision_), 
                 np.mean(val_macro_recall_), 
                 np.mean(val_macro_f1_), 
                 np.mean(val_accuracy_), 
                 np.mean(val_auc_)])))
        
    result = pd.DataFrame(result_)
    return(result)


#%%
from sklearn.ensemble import IsolationForest

isrf_auc = []
isrf_prec = []
isrf_recall = []
isrf_acc = []
isrf_f1 = []

for i in tqdm(range(50)):
    x_train, x_test, y_train, y_test = train_test_split(feature, y_true, test_size = 0.20, random_state = i)
    
    sm = SMOTE(random_state = i)
    x_train, y_train = sm.fit_sample(x_train, y_train)
    
    params_dict = {'random_state': [i],
                   'n_estimators': np.arange(50, 160, 10),
                   'max_features': [1, 5, 10]}
    params_grid = ParameterGrid(params_dict)
    
    cv_result = BinaryCV(
            pd.DataFrame(x_train), 
            pd.DataFrame(y_train), 
            IsolationForest,
            params_grid
    )

    max_recall_idx = cv_result.val_macro_recall.argmax(axis = 0)
    best_params = cv_result.iloc[max_recall_idx][:3].to_dict()
    best_params['random_state'] = int(i)

    isrf = IsolationForest(random_state = i)
    isrf.fit(x_train)
    
    # isrf_pred = isrf.predict(x_test)
    isrf_pred_prob = abs(isrf.score_samples(x_test))
    isrf_pred = [1 if i-0.5>0 else 0 for i in isrf_pred_prob]
    
    isrf_auc.append(roc_auc_score(y_test, isrf_pred_prob))
    isrf_prec.append(precision_score(y_test, isrf_pred))
    isrf_recall.append(recall_score(y_test, isrf_pred))
    isrf_acc.append(accuracy_score(y_test, isrf_pred))
    isrf_f1.append(f1_score(y_test, isrf_pred))


print('auc: ', np.mean(isrf_auc).round(3), '(', (np.std(isrf_auc, ddof = 1) / np.sqrt(len(isrf_auc))).round(3), ')',
      '\nprecision: ', np.mean(isrf_prec).round(3), '(', (np.std(isrf_prec, ddof = 1)/np.sqrt(len(isrf_prec))).round(3), ')',
      '\nrecall: ', np.mean(isrf_recall).round(3), '(', (np.std(isrf_recall, ddof = 1)/np.sqrt(len(isrf_recall))).round(3), ')',
      '\nacuracy: ', np.mean(isrf_acc).round(3), '(', (np.std(isrf_acc, ddof = 1)/np.sqrt(len(isrf_acc))).round(3) ,')',
      '\nf1: ', np.mean(isrf_f1).round(3), '(', (np.std(isrf_f1, ddof = 1)/np.sqrt(len(isrf_f1))).round(3), ')')


