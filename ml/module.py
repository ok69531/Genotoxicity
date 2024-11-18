import argparse

import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tg_num', type = int, default = 471, help = 'OECD TG for Genotoxicity (471, 473, 476, 487 / 474, 475, 478, 483, 486, 488)')
    parser.add_argument('--model', type = str, default = 'dt', help = 'dt, rf, gbt, xgb, lgb')
    parser.add_argument('--train_frac', type = float, default = 0.8, help = 'fraction of train dataset')
    parser.add_argument('--val_frac', type = float, default = 0.1, help = 'fraction of validation and test dataset')
    
    try: args = parser.parse_args()
    except: args = parser.parse_args([])
    
    return args


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
        clf = GradientBoostingClassifier(random_state = seed, **{k: v for k, v in param.items() if k != 'class_weight'})
    
    elif model == 'xgb':
        clf = XGBClassifier(random_state = seed, **{k: v for k, v in param.items() if k != 'class_weight'})
    
    elif model == 'lgb':
        clf = LGBMClassifier(random_state = seed, **param)
        
    return clf
