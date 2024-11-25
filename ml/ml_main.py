import os
import json
import warnings
import logging

import torch
from torch.utils.data import random_split

import numpy as np
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

from module import get_args, load_dataset, load_model
from get_params_comb import load_hyperparameters

warnings.filterwarnings('ignore')
logging.basicConfig(format='', level=logging.INFO)


args = get_args()

def main():
    x, y, fp_length = load_dataset(args)

    params = load_hyperparameters(args.model, args.tg_num)
    results_dict = {
        i: {'param': p,
            'valid': {'f1': list(), 'precision': list(), 'recall': list(), 'acc': list(), 'auc': list()},
            'test': {'f1': list(), 'precision': list(), 'recall': list(), 'acc': list(), 'auc': list()}}
        for i, p in enumerate(params)
    }

    if args.tg_num == 471:
        seeds = [8, 14, 51, 79, 123, 132, 139, 161, 201, 280]
    elif args.tg_num == 473:
        seeds = []
    elif args.tg_num == 476:
        seeds = []
    elif args.tg_num == 487:
        seed = []
    elif args.tg_num == 474:
        seeds = []
    elif args.tg_num == 475:
        seeds = []
    elif args.tg_num == 478:
        seeds = []
    
    # if args.tg_num == 471:
    #     seeds = [8, 14, 51, 79, 123, 132, 139, 161, 201, 280]
    # elif args.tg_num == 473:
    #     seeds = [48, 76, 214, 222, 424, 475, 550, 563, 634, 731]
    # elif args.tg_num == 476:
    #     seeds = [174, 752, 1224, 1378, 1448, 1545, 2042, 2147, 2362, 3554]
    # elif args.tg_num == 487:
    #     seed = [17, 28, 122, 173, 189, 206, 209, 225, 245, 268]
    # elif args.tg_num == 474:
    #     seeds = [322, 1190, 1485, 1747, 1915, 2509, 3184, 3720, 4371, 5087]
    # elif args.tg_num == 475:
    #     seeds = [17, 21, 113, 229, 238, 240, 245, 272, 295, 372]
    # elif args.tg_num == 478:
    #     seeds = [4, 8, 16, 33, 38, 41, 47, 63, 68, 74]
    # elif args.tg_num == 486:
    #     seeds = []
    
    for seed in seeds:
        logging.info('==================== Seed: {} ===================='.format(seed))
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
        
        for i, p in tqdm(enumerate(params)):
            if (args.model == 'gbt') & (p['class_weight'] is not None):
                model = load_model(args.model, seed, p)
                sample_weights = compute_sample_weight(p['class_weight'], y_train)
                model.fit(x_train, y_train, sample_weight=sample_weights)
            
            else:
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
    
    save_path = f'saved_result/tg{args.tg_num}'
    if os.path.isdir(save_path):
        pass
    else:
        os.makedirs(save_path)
    
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
    json.dump(best_result, open(save_path, 'w'))
    
    logging.info('')
    logging.info('Model: {}'.format(args.model))
    logging.info('TG: {}'.format(args.tg_num))
    logging.info('Target: {}'.format(args.target))
    logging.info('')
    logging.info('Fingerprint: {}'.format(args.fp_type))
    logging.info('Use Descriotors: {}'.format(args.use_md))
    logging.info('SMOTE: {}'.format(args.use_smote) )
    logging.info('')
    logging.info('param: {}'.format(param))
    logging.info('test f1: ${{{:.3f}}}_{{\\pm {:.3f}}}$'.format(np.mean(test_f1s) * 100, np.std(test_f1s) * 100))
    logging.info('test precision: ${{{:.3f}}}_{{\\pm {:.3f}}}$'.format(np.mean(test_precs) * 100, np.std(test_precs) * 100))
    logging.info('test recall: ${{{:.3f}}}_{{\\pm {:.3f}}}$'.format(np.mean(test_recs) * 100, np.std(test_recs) * 100))
    logging.info('test accuracy: ${{{:.3f}}}_{{\\pm {:.3f}}}$'.format(np.mean(test_accs) * 100, np.std(test_accs) * 100))
    logging.info('test roc-auc: ${{{:.3f}}}_{{\\pm {:.3f}}}$'.format(np.mean(test_aucs) * 100, np.std(test_aucs) * 100))
            

if __name__ == '__main__':
    main()