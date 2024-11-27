import os
import json
import warnings
import logging

import torch
from torch.utils.data import random_split

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    f1_score,
    classification_report
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_sample_weight

from imblearn.over_sampling import SMOTE

from module import get_args, get_seed, get_fingerprint, get_descriptors, load_model

warnings.filterwarnings('ignore')
logging.basicConfig(format='', level=logging.INFO)


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
    
    return df, x, y, fp_length


args = get_args()

def main():
    logging.info('Loading Dataset')
    df, x, y, fp_length = load_dataset(args)
    df = df[['Chemical', 'CasRN', args.target]]
    df[args.target] = y

    params_dict = {}
    for args.model in ['dt' ,'rf', 'gbt', 'xgb', 'lgb']:
        save_path = f'saved_result/tg{args.tg_num}'
        if args.use_smote:
            if args.use_md:
                save_path = os.path.join(save_path, f'{args.target}_tg{args.tg_num}_{args.fp_type}_md_smote_{args.model}.json')
            else:
                save_path = os.path.join(save_path, f'{args.target}_tg{args.tg_num}_{args.fp_type}_smote_{args.model}.json')
        else:
            if args.use_md:
                save_path = os.path.join(save_path, f'{args.target}_tg{args.tg_num}_{args.fp_type}_md_{args.model}.json')
            else:
                save_path = os.path.join(save_path, f'{args.target}_tg{args.tg_num}_{args.fp_type}_{args.model}.json')
        
        with open(save_path, 'r') as f:
            results_dict = json.load(f)
        
        params_dict[args.model] = results_dict['param']
    
    seeds = get_seed(args.tg_num)
    
    test_results_dict = {f'seed{i}': None for i in range(len(seeds))}
    
    for seed in seeds:
        logging.info('==================== Seed: {} ===================='.format(seeds.index(seed)))
        torch.manual_seed(seed)
        
        if (args.tg_num == 475) or (args.tg_num == 478) or (args.tg_num == 486):
            args.train_frac = 0.7
        
        num_train = int(len(x) * args.train_frac)
        num_valid = int(len(x) * args.val_frac)
        num_test = len(x) - (num_train + num_valid)
        assert num_train + num_valid + num_test == len(x)

        indices = torch.arange(len(x))
        train_idx, val_idx, test_idx = random_split(indices, [num_train, num_valid, num_test])

        x_train = x[np.array(train_idx)]; y_train = y[np.array(train_idx)]
        x_val = x[np.array(val_idx)]; y_val = y[np.array(val_idx)]
        x_test = x[np.array(test_idx)]; y_test = y[np.array(test_idx)]
        test_df = df.iloc[list(test_idx)]
        
        if args.use_md:
            scaler = MinMaxScaler()
            scaled_train_descriptors = scaler.fit_transform(x_train[:, fp_length:], y_train)
            x_train[:, fp_length:] = scaled_train_descriptors
            
            scaled_val_descriptors = scaler.transform(x_val[:, fp_length:])
            x_val[:, fp_length:] = scaled_val_descriptors
            
            scaled_test_descriptors = scaler.transform(x_test[:, fp_length:])
            x_test[:, fp_length:] = scaled_test_descriptors
            
        if args.use_smote:
            smote = SMOTE(random_state = args.smote_seed)
            x_train, y_train = smote.fit_resample(x_train, y_train)
        
        for args.model in ['dt' ,'rf', 'gbt', 'xgb', 'lgb']:
            params = params_dict[args.model]
            
            if (args.model == 'gbt') & (params['class_weight'] is not None):
                model = load_model(args.model, seed, params)
                sample_weights = compute_sample_weight(params['class_weight'], y_train)
                model.fit(x_train, y_train, sample_weight=sample_weights)
            
            else:
                model = load_model(args.model, seed, params)
                model.fit(x_train, y_train)
            
            test_pred = model.predict(x_test)
            test_pred_prob = model.predict_proba(x_test)[:, 1]
            
            test_df[f'{args.model}_pred'] = test_pred
            test_df[f'{args.model}_pred_prob'] = test_pred_prob
        
        test_results_dict[f'seed{seeds.index(seed)}'] = test_df
    
    
    save_path = f'saved_test_pred'
    if os.path.isdir(save_path):
        pass
    else:
        os.makedirs(save_path)
    
    if args.use_smote:
        if args.use_md:
            save_path = os.path.join(save_path, f'{args.target}_tg{args.tg_num}_{args.fp_type}_md_smote.xlsx')
        else:
            save_path = os.path.join(save_path, f'{args.target}_tg{args.tg_num}_{args.fp_type}_smote.xlsx')
    else:
        if args.use_md:
            save_path = os.path.join(save_path, f'{args.target}_tg{args.tg_num}_{args.fp_type}_md.xlsx')
        else:
            save_path = os.path.join(save_path, f'{args.target}_tg{args.tg_num}_{args.fp_type}.xlsx')
    
    writer = pd.ExcelWriter(save_path, engine='xlsxwriter', mode = 'w')
    for name, data in test_results_dict.items():
        data.to_excel(writer, sheet_name = name, index = False)
    writer.close()


if __name__ == '__main__':
    main()
