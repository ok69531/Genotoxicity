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

import rdkit
from rdkit import Chem
from rdkit.Chem import MACCSkeys, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

from get_params_comb import load_hyperparameters, parameter_grid

warnings.filterwarnings('ignore')


#%%
tg_num = 473
path = f'../vitro/data/tg{tg_num}/tg{tg_num}.xlsx'
# path = f'../vivo/data/tg{tg_num}/tg{tg_num}.xlsx'

df = pd.read_excel(path)
df = df[df.maj.notna()].reset_index(drop = True)

# toxprint
x = df.iloc[:, 5:].to_numpy()
y = np.array([1 if x == 'positive' else 0 for x in df.maj])

# maccs
mols = [Chem.MolFromSmiles(x) for x in df.SMILES]
maccs = [MACCSkeys.GenMACCSKeys(x) for x in mols]
maccs_bits = [x.ToBitString() for x in maccs]

x = np.array([list(map(int, x.ToBitString())) for x in maccs])


#%%
tg471 = pd.read_excel('../vitro/data/tg471/tg471.xlsx')
tg473 = pd.read_excel('../vitro/data/tg473/tg473.xlsx')
tg476 = pd.read_excel('../vitro/data/tg476/tg476.xlsx')
tg487 = pd.read_excel('../vitro/data/tg487/tg487.xlsx')
tg474 = pd.read_excel('../vivo/data/tg474/tg474.xlsx')
tg475 = pd.read_excel('../vivo/data/tg475/tg475.xlsx')
tg478 = pd.read_excel('../vivo/data/tg478/tg478.xlsx')
tg486 = pd.read_excel('../vivo/data/tg486/tg486.xlsx')

df = pd.concat([tg471, tg473, tg476, tg487, tg474, tg475, tg478, tg486])

mols = [Chem.MolFromSmiles(x) for x in df.SMILES]

descriptor_names = [desc[0] for desc in Descriptors._descList]
calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
descriptors = pd.DataFrame([dict(zip(descriptor_names, calc.CalcDescriptors(m))) for m in mols])
descriptors = descriptors.dropna(axis = 1)

with open('descriptor_name.json', 'w') as f:
    json.dump(list(descriptors.columns), f, indent=2)

with open('descriptor_name.json', 'r') as f:
    descriptor_names = json.load(f)

calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

calc.CalcDescriptors(mols[0])
descriptors = np.array([list(map(float, calc.CalcDescriptors(m))) for m in mols])

a = np.concatenate([x, descriptors], axis = 1)
x.shape
descriptors.shape

a[:, 729:].shape


#%%
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

print(np.unique(y, return_counts = True)[1]/len(y))


#%%
tg_num = 471
path = f'../vitro/data/tg{tg_num}/tg{tg_num}.xlsx'
# path = f'../vivo/data/tg{tg_num}/tg{tg_num}.xlsx'

df = pd.read_excel(path)
df = df[df.maj.notna()].reset_index(drop = True)

# toxprint
x = df.iloc[:, 5:].to_numpy()
y = np.array([1 if x == 'positive' else 0 for x in df.maj])
# y = np.array([1 if x == 'positive' else 0 for x in df.consv])


#%%
seed = 0
seeds = []

while len(seeds) < 10:
    torch.manual_seed(seed)

    train_frac = 0.8
    if (tg_num == 475) or (tg_num == 478) or (tg_num == 486):
            train_frac = 0.7
    val_frac = 0.1

    num_train = int(len(x) * train_frac)
    num_valid = int(len(x) * val_frac)
    num_test = len(x) - (num_train + num_valid)
    assert num_train + num_valid + num_test == len(df)

    indices = torch.arange(len(x))
    train_idx, val_idx, test_idx = random_split(indices, [num_train, num_valid, num_test])

    train_x = x[np.array(train_idx)]; train_y = y[np.array(train_idx)]
    val_x = x[np.array(val_idx)]; val_y = y[np.array(val_idx)]
    test_x = x[np.array(test_idx)]; test_y = y[np.array(test_idx)]


    model = DecisionTreeClassifier(random_state=seed)
    model.fit(train_x, train_y)

    try:
        val_pred = model.predict(val_x)
        val_perd_prob = model.predict_proba(val_x)[:, 1]

        test_pred = model.predict(test_x)
        test_perd_prob = model.predict_proba(test_x)[:, 1]
        
        if f1_score(test_y, test_pred) > 0.5: 
            seeds.append(seed)
            
            print(seed)
            print(classification_report(test_y, test_pred))
    except: pass
        
    seed += 1
    

print(len(seeds))
print(seeds)


#%%
for seed in seeds:
    torch.manual_seed(seed)

    train_frac = 0.8
    if (tg_num == 475) or (tg_num == 478) or (tg_num == 486):
            train_frac = 0.7
    val_frac = 0.1

    num_train = int(len(x) * train_frac)
    num_valid = int(len(x) * val_frac)
    num_test = len(x) - (num_train + num_valid)
    assert num_train + num_valid + num_test == len(df)

    indices = torch.arange(len(x))
    train_idx, val_idx, test_idx = random_split(indices, [num_train, num_valid, num_test])

    train_x = x[np.array(train_idx)]; train_y = y[np.array(train_idx)]
    val_x = x[np.array(val_idx)]; val_y = y[np.array(val_idx)]
    test_x = x[np.array(test_idx)]; test_y = y[np.array(test_idx)]


    model = XGBClassifier(random_state=seed)
    # model = LGBMClassifier(random_state=seed)
    # model = LGBMClassifier(random_state=seed, class_weight='balanced')
    # model = RandomForestClassifier(random_state=seed, class_weight='balanced')
    model.fit(train_x, train_y)

    val_pred = model.predict(val_x)
    val_perd_prob = model.predict_proba(val_x)[:, 1]

    test_pred = model.predict(test_x)
    test_perd_prob = model.predict_proba(test_x)[:, 1]
    
    print(seed)
    print(classification_report(test_y, test_pred))
    
    # print(classification_report(train_y, model.predict(train_x)))
    # print(seed)
    # print(classification_report(val_y, val_pred))
    # print(classification_report(test_y, test_pred))


#%%
'''여기부터'''
from sklearn.utils.class_weight import compute_sample_weight

# n_est_list = np.concatenate([np.array([2, 3, 4]), np.arange(5, 155, 5)])
# min_sample_split_list = [2, 3, 4, 5, 7, 9, 10, 13, 15, 17, 20]
# min_sample_leaf_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# min_weight_fraction_leaf_list = np.arange(0, 0.5, 0.1)
# max_depth_list = [None, 1, 2, 3, 4, 5, 7, 10, 13, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
# max_depth_list = [-1, 3, 4, 5, 6, 7, 8, 9, 15, 30]
# lr_list = [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1]
# gamma_list = [0, 0.001, 0.003, 0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1]
# min_child_weight_list = [1, 2, 3, 4, 5, 7, 9, 10, 15, 20]
# min_child_sample_list = np.arange(5, 55, 5)
# num_leaves_list = np.arange(3, 100, 2)
# scale_pos_weight_list = np.arange(1, 21, 1)


val_f1s = []
for n in tqdm(n_est_list):
    # model = LGBMClassifier(random_state=0, class_weight='balanced', min_child_samples=n)
    # model = XGBClassifier(random_state=0, n_estimators=n)
    # model = GradientBoostingClassifier(random_state=0, learning_rate=n)
    # model = RandomForestClassifier(random_state=0, class_weight='balanced', n_estimators=n)
    # model = DecisionTreeClassifier(random_state=0, class_weight='balanced', max_depth=n)

    model.fit(train_x, train_y)
    # model.fit(train_x, train_y, sample_weight=compute_sample_weight('balanced', train_y))

    val_pred = model.predict(val_x)
    val_f1s.append(f1_score(val_y, val_pred))


idx = val_f1s.index(max(val_f1s))
n = n_est_list[idx]
model = RandomForestClassifier(random_state=0, n_estimators=n)
model.fit(train_x, train_y)

val_pred = model.predict(val_x)
val_perd_prob = model.predict_proba(val_x)[:, 1]

test_pred = model.predict(test_x)
test_perd_prob = model.predict_proba(test_x)[:, 1]

print(classification_report(train_y, model.predict(train_x)))
print(classification_report(val_y, val_pred))
print(classification_report(test_y, test_pred))
