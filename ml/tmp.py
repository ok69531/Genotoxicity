#%%
import os
import json
import argparse
from random import Random
import warnings

import torch
from torch.utils.data import random_split

import numpy as np
import pandas as pd
from tqdm import tqdm

# from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    f1_score,
    classification_report
)

from get_params_comb import load_hyperparameters, parameter_grid

warnings.filterwarnings('ignore')


#%%
tg_num = 476
path = f'../vitro/data/tg{tg_num}/tg{tg_num}.xlsx'

df = pd.read_excel(path)
x = df.iloc[:, 5:].to_numpy()
y = np.array([1 if x == 'positive' else 0 for x in df.maj])

seed = 0
torch.manual_seed(seed)
 
num_valid = int(len(df) * 0.1)
num_test = int(len(df) * 0.1)
num_train = len(df) - (num_valid + num_test)
assert num_train + num_valid + num_test == len(df)

indices = torch.arange(len(x))
train_idx, val_idx, test_idx = random_split(indices, [num_train, num_valid, num_test])

train_x = x[np.array(train_idx)]; train_y = y[np.array(train_idx)]
val_x = x[np.array(val_idx)]; val_y = y[np.array(val_idx)]
test_x = x[np.array(test_idx)]; test_y = y[np.array(test_idx)]

print(np.unique(train_y, return_counts = True)[1]/len(train_x))
print(np.unique(val_y, return_counts = True)[1]/len(val_x))
print(np.unique(test_y, return_counts = True)[1]/len(test_x))

print(np.unique(y, return_counts = True)[1]/len(test_x))


#%%
model = DecisionTreeClassifier(random_state=0)
model = DecisionTreeClassifier(random_state=0, class_weight='balanced')
model = DecisionTreeClassifier(random_state=0, class_weight={0: 0.9, 1: 0.1})
model.fit(train_x, train_y)

from sklearn.utils.class_weight import compute_sample_weight
compute_sample_weight(class_weight='balanced', y=train_y)

param
p = {k: v for k, v in param.items() if k != 'class_weight'}
GradientBoostingClassifier()


#%%
'''여기부터'''
# n_est_list = np.concatenate([np.array([2, 3, 4]), np.arange(5, 155, 5)])
# min_sample_split_list = [2, 3, 4, 5, 7, 9, 10, 13, 15, 17, 20]
# min_sample_leaf_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# max_depth_list = [None, 1, 2, 3, 4, 5, 7, 10, 13, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
# max_depth_list = [-1, 3, 4, 5, 6, 7, 8, 9, 15, 30]
# lr_list = [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1]
# gamma_list = [0, 0.001, 0.003, 0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1]
# min_child_weight_list = [1, 2, 3, 4, 5, 7, 9, 10, 15, 20]
# num_leaves_list = np.arange(3, 100, 2)


val_f1s = []
for n in tqdm(max_depth_list):
    # model = LGBMClassifier(random_state=0, n_estimators=n)
    # model = XGBClassifier(random_state=0, min_child_weight=n)
    # model = GradientBoostingClassifier(random_state=0, max_depth=n)
    # model = DecisionTreeClassifier(random_state=0, min_samples_leaf=n)
    # model = RandomForestClassifier(random_state=0, max_depth=n, n_estimators=5)

    model.fit(train_x, train_y)

    val_pred = model.predict(val_x)
    val_f1s.append(f1_score(val_y, val_pred))


idx = val_f1s.index(max(val_f1s))
n = max_depth_list[idx]
model = DecisionTreeClassifier(random_state=0, min_samples_leaf=n)
model.fit(train_x, train_y)

val_pred = model.predict(val_x)
val_perd_prob = model.predict_proba(val_x)[:, 1]

test_pred = model.predict(test_x)
test_perd_prob = model.predict_proba(test_x)[:, 1]

print(classification_report(train_y, model.predict(train_x)))
print(classification_report(val_y, val_pred))
print(classification_report(test_y, test_pred))


#%%

def get_471_params(model: str):
    if model == 'dt':
        params_dict = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 25, 30, 35, 40, 45, 50, 55],
            'min_samples_split': [2, 3, 4, 5, 10],
            'min_samples_leaf': [1, 2, 5],
            'class_weight': [None, 'balanced']
        }

    elif model == 'rf':
        params_dict = {
            'n_estimators': [10, 15, 30, 50, 70, 90, 100, 110, 130, 150],
            'min_samples_split': [2, 3, 4, 5],
            'min_samples_leaf': [1, 2],
            'max_depth': [None, 25, 30, 35, 40, 50],
            'class_weight': [None, 'balanced']
        }
    
    elif model == 'gbt':
        params_dict = {
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'n_estimators': [10, 20, 30, 50, 70, 100, 130, 150],
            'max_depth': [None, 2, 3, 4],
            'min_samples_split': [2, 3, 4, 5],
            'class_weight': [None, 'balanced']
            }
    
    elif model == 'xgb':
        params_dict = {
            'n_estimators': [10, 20, 30, 50, 100],
            'learning_rate': [0.05, 0.1],
            'min_child_weight': [1, 3],
            'max_depth': [3, 6, 9],
            'gamma': [0, 0.001, 0.005, 0.01, 0.1, 1],
        }
        
    elif model == 'lgb':
        params_dict = {
            'num_leaves': [15, 21, 31, 33, 39, 50, 70, 99],
            'max_depth': [-1, 3, 5, 8],
            'n_estimators': [100, 110],
            'min_child_samples': [10, 20, 25, 30],
            'class_weight': [None, 'balanced']
        }
    
    return params_dict

param = parameter_grid(params_dict)[0]
len(parameter_grid(params_dict))


#%%
parser = argparse.ArgumentParser()
parser.add_argument('--tg_num', type = int, default = 471, help = 'OECD TG for Genotoxicity (471, 473, 476, 487 / 474, 475, 478, 483, 486, 488)')
parser.add_argument('--model', type = str, default = 'dt', help = 'dt, rf, gbt, xgb, lgb')
parser.add_argument('--train_frac', type = float, default = 0.8, help = 'fraction of train dataset')
parser.add_argument('--val_frac', type = float, default = 0.1, help = 'fraction of validation and test dataset')
try: args = parser.parse_args()
except: args = parser.parse_args([])


def load_dataset(tg: int):
    path = f'../vitro/data/tg{tg}/tg{tg}.xlsx'
    df = pd.read_excel(path)
    
    x = df.iloc[:, 5:].to_numpy()
    y = np.array([1 if x == 'positive' else 0 for x in df.maj])
    
    return x, y


def load_model(model: str, seed: int, param: dict):
    if model == 'dt':
        clf = DecisionTreeClassifier(random_state = seed, **param)
    
    elif model == 'rf':
        clf = RandomForestClassifier(random_state = seed, **param)
    
    elif model == 'gbt':
        clf = GradientBoostingClassifier(random_state = seed, **param)
    
    elif model == 'xgb':
        clf = XGBClassifier(random_state = seed, **param)
    
    elif model == 'lgb':
        clf = LGBMClassifier(random_state = seed, **param)
        
    return clf



x, y = load_dataset(args.tg_num)

params = load_hyperparameters(args.model, args.tg_num)
results_dict = {
    i: {'param': p,
        'valid': {'f1': list(), 'precision': list(), 'recall': list(), 'acc': list(), 'auc': list()},
        'test': {'f1': list(), 'precision': list(), 'recall': list(), 'acc': list(), 'auc': list()}}
    for i, p in enumerate(params)
}

for seed in range(10):
    torch.manual_seed(seed)
    
    num_train = int(len(x) * args.train_frac)
    num_valid = int(len(x) * args.val_frac)
    num_test = len(x) - (num_train + num_valid)
    assert num_train + num_valid + num_test == len(x)

    indices = torch.arange(len(x))
    train_idx, val_idx, test_idx = random_split(indices, [num_train, num_valid, num_test])

    x_train = x[np.array(train_idx)]; y_train = y[np.array(train_idx)]
    x_val = x[np.array(val_idx)]; y_val = y[np.array(val_idx)]
    x_test = x[np.array(test_idx)]; y_test = y[np.array(test_idx)]
    
    for i, p in enumerate(params):
        model = load_model(args.model, seed, p)
        model.fit(x_train, y_train)
        
        valid_pred = model.predict(x_val)
        valid_pred_prob = model.predict_proba(x_val)[:, 1]
        
        test_pred = model.predict(x_test)
        test_pred_prob = model.predict_proba(x_test)[:, 1]
        
        results_dict[i]['valid']['f1'].append(f1_score(y_val, valid_pred))
        results_dict[i]['valid']['precision'].append(precision_score(y_val, valid_pred))
        results_dict[i]['valid']['recall'].append(recall_score(y_val, valid_pred))
        results_dict[i]['valid']['acc'].append(accuracy_score(y_val, valid_pred))
        results_dict[i]['valid']['auc'].append(roc_auc_score(y_val, valid_pred_prob))
        
        results_dict[i]['test']['f1'].append(f1_score(y_test, test_pred))
        results_dict[i]['test']['precision'].append(precision_score(y_test, test_pred))
        results_dict[i]['test']['recall'].append(recall_score(y_test, test_pred))
        results_dict[i]['test']['acc'].append(accuracy_score(y_test, test_pred))
        results_dict[i]['test']['auc'].append(roc_auc_score(y_test, test_pred_prob))
        
    val_f1s = [np.mean(results_dict[i]['valid']['f1']) for i in range(len(params))]
    max_idx = val_f1s.index(max(val_f1s))
    
    best_result = results_dict[max_idx] 
    param = best_result['param']
    test_f1s = best_result['test']['f1']
    test_precs = best_result['test']['precision']
    test_recs = best_result['test']['recall']
    test_accs = best_result['test']['acc']
    test_aucs = best_result['test']['auc']
    
    save_path = 'saved_result'
    if os.path.isdir(save_path):
        pass
    else:
        os.makedirs(save_path)
        
    save_path = os.path.join(save_path, f'tg{args.tg_num}_{args.model}.json')
    json.dump(best_result, open(save_path, 'w'))
    
    print(f'param: {param}')
    print(f'test f1: ${{{np.mean(test_f1s)*100:.3f}}}_{{\pm {np.std(test_f1s)*100:.3f}}}$')
    print(f'test precision: ${{{np.mean(test_precs)*100:.3f}}}_{{\pm {np.std(test_precs)*100:.3f}}}$')
    print(f'test recall: ${{{np.mean(test_recs)*100:.3f}}}_{{\pm {np.std(test_recs)*100:.3f}}}$')
    print(f'test accuracy: ${{{np.mean(test_accs)*100:.3f}}}_{{\pm {np.std(test_accs)*100:.3f}}}$')
    print(f'test roc-auc: ${{{np.mean(test_aucs)*100:.3f}}}_{{\pm {np.std(test_aucs)*100:.3f}}}$')
    