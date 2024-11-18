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
from sklearn.utils.class_weight import compute_sample_weight

from module import get_args, load_dataset, load_model
from get_params_comb import load_hyperparameters

warnings.filterwarnings('ignore')
logging.basicConfig(format='', level=logging.INFO)

args = get_args()

def main():
    x, y = load_dataset(args.tg_num)

    params = load_hyperparameters(args.model, args.tg_num)
    results_dict = {
        i: {'param': p,
            'valid': {'f1': list(), 'precision': list(), 'recall': list(), 'acc': list(), 'auc': list()},
            'test': {'f1': list(), 'precision': list(), 'recall': list(), 'acc': list(), 'auc': list()}}
        for i, p in enumerate(params)
    }

    for seed in range(10):
        logging.info('==================== Seed: {} ===================='.format(seed))
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
    
    save_path = 'saved_result'
    if os.path.isdir(save_path):
        pass
    else:
        os.makedirs(save_path)
        
    save_path = os.path.join(save_path, f'tg{args.tg_num}_{args.model}.json')
    json.dump(best_result, open(save_path, 'w'))
    
    logging.info('')
    logging.info('Model: {}'.format(args.model))
    logging.info('TG: {}'.format(args.tg_num))
    logging.info('param: {}'.format(param))
    logging.info('test f1: ${{{:.3f}}}_{{\\pm {:.3f}}}$'.format(np.mean(test_f1s) * 100, np.std(test_f1s) * 100))
    logging.info('test precision: ${{{:.3f}}}_{{\\pm {:.3f}}}$'.format(np.mean(test_precs) * 100, np.std(test_precs) * 100))
    logging.info('test recall: ${{{:.3f}}}_{{\\pm {:.3f}}}$'.format(np.mean(test_recs) * 100, np.std(test_recs) * 100))
    logging.info('test accuracy: ${{{:.3f}}}_{{\\pm {:.3f}}}$'.format(np.mean(test_accs) * 100, np.std(test_accs) * 100))
    logging.info('test roc-auc: ${{{:.3f}}}_{{\\pm {:.3f}}}$'.format(np.mean(test_aucs) * 100, np.std(test_aucs) * 100))
            

if __name__ == '__main__':
    main()