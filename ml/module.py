import json
import argparse

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors

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
    parser.add_argument('--target', type = str, default = 'maj', help = 'maj or consv')
    parser.add_argument('--fp_type', type = str, default = 'toxprint', help = 'type of fingerprints (toxprint, maccs, topology, morgan, rdkit)')
    parser.add_argument('--train_frac', type = float, default = 0.8, help = 'fraction of train dataset')
    parser.add_argument('--val_frac', type = float, default = 0.1, help = 'fraction of validation and test dataset')
    parser.add_argument('--use_smote', type = bool, default = False, help = 'whether using SMOTE')
    parser.add_argument('--smote_seed', type = int, default = 42)
    parser.add_argument('--use_md', type = bool, default = False, help = 'whetehr using molecular descriptors')
    
    try: args = parser.parse_args()
    except: args = parser.parse_args([])
    
    return args


def load_dataset(args):
    tg = args.tg_num
    fp_type = args.fp_type
    use_md = args.use_md
    target = args.target
    
    if (tg == 471) or (tg == 473) or (tg == 476) or (tg == 487):
        path = f'../vitro/data/tg{tg}/tg{tg}.xlsx'
    else: 
        path = f'../vivo/data/tg{tg}/tg{tg}.xlsx'
    df = pd.read_excel(path)
    
    if target == 'maj':
        df = df[df.maj.notna()].reset_index(drop = True)
        y = np.array([1 if x == 'positive' else 0 for x in df.maj])
    else:
        y = np.array([1 if x == 'positive' else 0 for x in df.consv])
        
    x = get_fingerprint(df, fp_type)
    fp_length = x.shape[1]
    if use_md:
        descriptors = get_descriptors(df)
        x = np.concatenate([x, descriptors], axis = 1)
    
    return x, y, fp_length


def get_fingerprint(df, fp_type: str):
    if fp_type not in ['toxprint', 'maccs', 'topology', 'morgan', 'rdkit']:
        raise ValueError('fingerprint %s not supported' % fp_type)
    
    if fp_type == 'toxprint':
        x = df.iloc[:, 5:].to_numpy()
    
    elif fp_type == 'maccs':
        mols = [Chem.MolFromSmiles(x) for x in df.SMILES]
        maccs = [MACCSkeys.GenMACCSKeys(x) for x in mols]
        x = np.array([list(map(int, x.ToBitString())) for x in maccs])

    return x


def get_descriptors(df):
    with open('descriptor_name.json', 'r') as f:
        descriptor_names = json.load(f)
    
    mols = [Chem.MolFromSmiles(x) for x in df.SMILES]
    
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    descriptors = np.array([list(map(float, calc.CalcDescriptors(m))) for m in mols])
    
    return descriptors


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
