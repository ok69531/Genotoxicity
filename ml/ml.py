#%%
import json
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from imblearn.over_sampling import SMOTE, SVMSMOTE

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold, 
    StratifiedShuffleSplit
)
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    f1_score,
    classification_report
)

from itertools import product
from collections.abc import Iterable

from rdkit import Chem
from rdkit.Chem import MACCSkeys

warnings.filterwarnings('ignore')


#%%
def load_model(model: str, seed: int, param: dict):
    if model == 'logistic':
        clf = LogisticRegression(random_state = seed, **param)
        
    elif model == 'dt':
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


def parameter_grid(param_dict):
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
def binary_cross_validation(model, x, y, seed, smote = False):
    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
    
    metrics = ['precision', 'recall', 'f1', 'accuracy', 'auc']
    
    train_metrics = list(map(lambda x: 'train_' + x, metrics))
    val_metrics = list(map(lambda x: 'val_' + x, metrics))
    
    train_precision = []
    train_recall = []
    train_f1 = []
    train_accuracy = []
    train_auc = []
    
    val_precision = []
    val_recall = []
    val_f1 = []
    val_accuracy = []
    val_auc = []
    
    for train_idx, val_idx in skf.split(x, y):
        train_x, train_y = x[train_idx], y[train_idx]
        val_x, val_y = x[val_idx], y[val_idx]
        
        if smote:
            sm = SMOTE(random_state = seed)
            train_x, train_y = sm.fit_resample(train_x, train_y)
        else:
            pass
        
        model.fit(train_x, train_y)
        
        train_pred = model.predict(train_x)
        train_pred_score = model.predict_proba(train_x)[:, 1]
        
        val_pred = model.predict(val_x)
        val_pred_score = model.predict_proba(val_x)[:, 1]
        
        train_precision.append(precision_score(train_y, train_pred))
        train_recall.append(recall_score(train_y, train_pred))
        train_f1.append(f1_score(train_y, train_pred))
        train_accuracy.append(accuracy_score(train_y, train_pred))
        train_auc.append(roc_auc_score(train_y, train_pred_score))

        val_precision.append(precision_score(val_y, val_pred))
        val_recall.append(recall_score(val_y, val_pred))
        val_f1.append(f1_score(val_y, val_pred))
        val_accuracy.append(accuracy_score(val_y, val_pred))
        val_auc.append(roc_auc_score(val_y, val_pred_score))

    result = dict(zip(train_metrics + val_metrics, 
                      [np.mean(train_precision), np.mean(train_recall), np.mean(train_f1), np.mean(train_accuracy), np.mean(train_auc), 
                       np.mean(val_precision), np.mean(val_recall), np.mean(val_f1), np.mean(val_accuracy), np.mean(val_auc)]))
    
    return(result)


def metric_mean(data, metric: str):
    mean_per_hp = list(map(lambda x: np.mean(x[1]), data[metric].items()))
    return mean_per_hp


def print_best_param(val_result, metric: str):
    
    mean_list = metric_mean(val_result, metric)
    max_idx = mean_list.index(max(mean_list))
    
    best_param = val_result['model'][f'model{max_idx}']
    
    return best_param


#%%
tg_num = 471
path = f'../vitro/data/tg{tg_num}/tg{tg_num}.xlsx'

df = pd.read_excel(path)
mols = [Chem.MolFromSmiles(x) for x in df.SMILES]
x = np.array([list(MACCSkeys.GenMACCSKeys(x))[1:] for x in mols])
y = np.array([1 if x == 'positive' else 0 for x in df.Genotoxicity_maj])

seed = 42
# seed = 858
num_valid = int(len(df) * 0.1)
num_test = int(len(df) * 0.1)
num_train = len(df) - (num_valid + num_test)
assert num_train + num_valid + num_test == len(df)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = num_test, random_state = seed)


#%%
model_type = 'rf' # logistic, dt, rf, gbt, lgb, xgb, 
num_run = 3
metric = 'f1'
smote = True

print('=================================')
print(f'tg{tg_num} majority {model_type} smote {smote}')

# cross validation
params = load_hyperparameter(model_type)

result = {}
result['model'] = {}
result['precision'] = {}
result['recall'] = {}
result['f1'] = {}
result['accuracy'] = {}
result['auc'] = {}


for p in tqdm(range(len(params))):
    
    result['model']['model'+str(p)] = params[p]
    result['precision']['model'+str(p)] = []
    result['recall']['model'+str(p)] = []
    result['f1']['model'+str(p)] = []
    result['accuracy']['model'+str(p)] = []
    result['auc']['model'+str(p)] = []
    
    for seed in range(num_run):
        model = load_model(model = model_type, seed = seed, param = params[p])
        
        cv_result = binary_cross_validation(model, x_train, y_train, seed, smote = smote)
        
        result['precision']['model'+str(p)].append(cv_result['val_precision'])
        result['recall']['model'+str(p)].append(cv_result['val_recall'])
        result['f1']['model'+str(p)].append(cv_result['val_f1'])
        result['accuracy']['model'+str(p)].append(cv_result['val_accuracy'])
        result['auc']['model'+str(p)].append(cv_result['val_auc'])

# json.dump(result, open(f'tg{args.tg_num}_val_results/binary/{args.inhale_type}_{args.model}.json', 'w'))

best_param = print_best_param(val_result = result, metric = metric)

m = list(result['model'].keys())[list(result['model'].values()).index(best_param)]

# val result
precision = result['precision'][m]
recall = result['recall'][m]
acc = result['accuracy'][m]
auc = result['auc'][m]
f1 = result['f1'][m]

print(f"best param: {best_param} \
        \nvalidation result \
        \nprecision: {np.mean(precision):.3f}({np.std(precision):.3f}) \
        \nrecall: {np.mean(recall):.3f}({np.std(recall):.3f}) \
        \naccuracy: {np.mean(acc):.3f}({np.std(acc):.3f}) \
        \nauc: {np.mean(auc):.3f}({np.std(auc):.3f}) \
        \nf1: {np.mean(f1):.3f}({np.std(f1):.3f})")

# test reulst
model = load_model(model = model_type, seed = seed, param = best_param)

model.fit(x_train, y_train)

pred = model.predict(x_test)
pred_score = model.predict_proba(x_test)[:, 1]
    
print(f'test result \
        \nbest param: {best_param} \
        \nprecision: {precision_score(y_test, pred):.3f} \
        \nrecall: {recall_score(y_test, pred):.3f} \
        \naccuracy: {accuracy_score(y_test, pred):.3f} \
        \nauc: {roc_auc_score(y_test, pred_score):.3f} \
        \nf1: {f1_score(y_test, pred):.3f}')


# %%
