from itertools import product
from collections.abc import Iterable


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


def load_hyperparameters(model: str, tg: int):
    if tg == 471:
        params_dict = get_471_params(model)
    # elif tg == 473
    
    params = parameter_grid(params_dict)
    
    return params


def get_471_params(model: str):
    if model == 'dt':
        params_dict = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 25, 30],
            'min_samples_split': [2, 3, 4, 5, 10],
            'min_samples_leaf': [1, 2, 5]
        }

    elif model == 'rf':
        params_dict = {
            'n_estimators': [10, 15, 30, 50, 70, 90, 100, 110, 130, 150],
            'min_samples_split': [2, 3, 4, 5],
            'min_samples_leaf': [1, 2],
            'max_depth': [None, 25, 30, 35, 40, 50]
        }
    
    elif model == 'gbt':
        params_dict = {
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'n_estimators': [10, 20, 30, 50, 70, 100, 130, 150],
            'max_depth': [2, 3, 4],
            'min_samples_split': [2, 3, 4, 5]
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
            'min_child_samples': [10, 20, 25, 30]
        }
    
    return params_dict
