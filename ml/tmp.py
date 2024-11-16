#%%
import os
import json
import argparse
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

from get_params_comb import load_hyperparameters

warnings.filterwarnings('ignore')


#%%


def load_hyperparameter(model: str):
    if model == 'logistic':
        params_dict = {
            'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                  1, 2, 3, 4, 5, 7, 9, 11, 15, 20, 25, 30, 35, 40, 50, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
        
    elif model == 'dt':
        params_dict = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 1, 2, 3, 4, 5],
            'min_samples_split': [2, 3, 4],
            'min_samples_leaf': [1, 2, 3]
        }
    
    elif model == 'rf':
        params_dict = {
            'n_estimators': [80, 90, 100, 110, 120, 130, 140, 150],
            'criterion': ['gini'],
            'min_samples_split': [2, 3, 4],
            'min_samples_leaf': [1, 2, 3, 4],
            'max_depth': [None, 2, 3, 4, 5],
            'max_features': ['sqrt', 'log2']
        }
    
    elif model == 'gbt':
        params_dict = {
            'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
            'n_estimators': [5, 10, 50, 100, 130],
            'max_depth': [1, 2, 3, 4]
        }
    
    elif model == 'xgb':
        params_dict = {
        'min_child_weight': [1, 2, 3, 5],
            'max_depth': [3, 6, 9],
            'gamma': np.linspace(0, 3, 10),
            # 'objective': ['multi:softmax'],
            'booster': ['gbtree']
        }
    
    elif model == 'lgb':
        params_dict = {
            # 'objective': ['multiclass'],
            'num_leaves': [15, 21, 27, 31, 33],
            'max_depth': [-1, 2],
            # 'n_estimators': [5, 10, 50, 100, 130],
            'min_child_samples': [10, 20, 25, 30]
        }
    
    params = parameter_grid(params_dict)
    
    return params



#%%

tg_num = 471
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


#%%
'''여기부터'''
# n_est_list = np.arange(5, 150, 5)
# min_sample_split_list = [2, 3, 4, 5, 10, 15, 20]
# min_sample_leaf_list = [1, 2, 5]
max_depth_list = [-1, 3, 4, 5, 6, 7, 8, 9, 15, 30]
# lr_list = [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1]
# gamma_list = [0, 0.001, 0.003, 0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1]
# min_child_weight_list = [1, 2, 3, 4, 5, 7, 9, 10, 15, 20]
# num_leaves_list = np.arange(15, 100, 2)


val_f1s = []
for n in tqdm(max_depth_list):
    model = LGBMClassifier(random_state=0, max_depth=n)
    # model = XGBClassifier(random_state=0, min_child_weight=n)
    # model = GradientBoostingClassifier(random_state=0, min_samples_split=n)
    # model = DecisionTreeClassifier(random_state=0, min_samples_split=n)
    # model = RandomForestClassifier(random_state=0, max_depth=n)
    model.fit(train_x, train_y)

    val_pred = model.predict(val_x)
    val_f1s.append(f1_score(val_y, val_pred))


idx = val_f1s.index(max(val_f1s))
n = max_depth_list[idx]
model = LGBMClassifier(random_state=0, num_leaves=n)
model.fit(train_x, train_y)

val_pred = model.predict(val_x)
val_perd_prob = model.predict_proba(val_x)[:, 1]

test_pred = model.predict(test_x)
test_perd_prob = model.predict_proba(test_x)[:, 1]

print(classification_report(train_y, model.predict(train_x)))
print(classification_report(val_y, val_pred))
print(classification_report(test_y, test_pred))


params_dict = {
    'num_leaves': [15, 21, 31, 33, 39, 50, 70, 99],
    'max_depth': [-1, 3, 5, 8],
    'n_estimators': [100, 110],
    'min_child_samples': [10, 20, 25, 30]
}
params = parameter_grid(params_dict)
len(params)


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
    