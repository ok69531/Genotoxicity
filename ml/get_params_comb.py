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
    elif tg == 473:
        params_dict = get_473_params(model)
    elif tg == 476:
        params_dict = get_476_params(model)
    elif tg == 487:
        params_dict = get_487_params(model)
    elif tg == 474:
        params_dict = get_474_params(model)
    elif tg == 475:
        params_dict = get_475_params(model)
    elif tg == 478:
        params_dict = get_478_params(model)
    
    params = parameter_grid(params_dict)
    
    return params


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
            'learning_rate': [0.03, 0.05, 0.1],
            'n_estimators': [10, 30, 50, 70, 100, 130, 150],
            'max_depth': [None, 2, 3, 4],
            'min_samples_split': [2, 3, 4, 5],
            'class_weight': [None, 'balanced']
            }
    
    elif model == 'xgb':
        params_dict = {
            'n_estimators': [10, 20, 30, 50, 100],
            'learning_rate': [0.05, 0.1],
            'min_child_weight': [1, 3],
            'max_depth': [None, 3, 6, 9],
            'gamma': [0, 0.001, 0.005, 0.01, 0.1, 1],
            'scale_pos_weight': [1, 3, 5, 10],
            'class_weight': [None]
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


def get_473_params(model: str):
    if model == 'dt':
        params_dict = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 25, 30, 35, 40, 45, 50, 55],
            'min_samples_split': [2, 3, 4, 5, 7, 10],
            'min_samples_leaf': [2, 3, 4, 5],
            'class_weight': [None, 'balanced']
        }

    elif model == 'rf':
        params_dict = {
            'n_estimators': [5, 10, 30, 50, 70, 90, 100, 110, 130, 150],
            'min_samples_split': [2, 3, 4, 5],
            'min_samples_leaf': [1, 2],
            'max_depth': [None, 25, 30, 35, 40, 45, 50],
            'class_weight': [None, 'balanced']
        }
    
    elif model == 'gbt':
        params_dict = {
            'learning_rate': [0.03, 0.05, 0.1],
            'n_estimators': [5, 10, 20, 30, 50, 70, 100, 130],
            'max_depth': [None, 3, 7, 10, 20],
            'min_samples_split': [2, 3, 4, 5],
            'class_weight': [None, 'balanced']
            }
    
    elif model == 'xgb':
        params_dict = {
            'n_estimators': [5, 10, 20, 30, 50, 100],
            'learning_rate': [0.05, 0.1],
            'min_child_weight': [1, 3],
            'max_depth': [None, 3, 6, 9],
            'gamma': [0, 0.001, 0.005, 0.01, 0.1, 1],
            'scale_pos_weight': [1, 3, 5, 10],
            'class_weight': [None]
        }
        
    elif model == 'lgb':
        params_dict = {
            'num_leaves': [15, 19, 21, 31, 33, 39, 43, 50, 67, 70, 99],
            'max_depth': [-1, 3, 5, 8],
            'n_estimators': [100, 110],
            'min_child_samples': [10, 20, 25, 30],
            'class_weight': [None, 'balanced']
        }
    
    return params_dict


def get_476_params(model: str):
    if model == 'dt':
        params_dict = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 25, 30, 35, 40, 45, 50, 55],
            'min_samples_split': [2, 3, 4, 5, 7, 10],
            'min_samples_leaf': [1, 2, 3, 4, 5],
            'class_weight': [None, 'balanced']
        }

    elif model == 'rf':
        params_dict = {
            'n_estimators': [5, 10, 15, 20, 25, 30, 50, 70, 90, 100, 110],
            'min_samples_split': [2, 3, 4, 5],
            'min_samples_leaf': [1, 2],
            'max_depth': [None, 25, 30, 35, 40, 45, 50],
            'class_weight': [None, 'balanced']
        }
    
    elif model == 'gbt':
        params_dict = {
            'learning_rate': [0.03, 0.05, 0.1],
            'n_estimators': [5, 10, 20, 30, 50, 70, 100, 130],
            'max_depth': [None, 10, 25, 35, 40],
            'min_samples_split': [2, 3, 4, 5],
            'class_weight': [None, 'balanced']
            }
    
    elif model == 'xgb':
        params_dict = {
            'n_estimators': [5, 10, 20, 30, 50, 100, 110, 120],
            'learning_rate': [0.05, 0.1],
            'min_child_weight': [1, 3],
            'max_depth': [None, 3, 6, 9],
            'gamma': [0, 0.001, 0.005, 0.01, 0.1, 1],
            'scale_pos_weight': [1, 3, 5, 10],
            'class_weight': [None]
        }
        
    elif model == 'lgb':
        params_dict = {
            'num_leaves': [11, 21, 25, 31, 33, 41, 43, 45, 50, 67, 70, 99],
            'max_depth': [-1, 2, 3, 5, 8],
            'n_estimators': [100, 110, 120],
            'min_child_samples': [10, 20, 25, 30],
            'class_weight': [None, 'balanced']
        }
    
    return params_dict


def get_487_params(model: str):
    if model == 'dt':
        params_dict = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 25, 30, 35, 40, 45, 50, 55],
            'min_samples_split': [2, 3, 4, 5, 7, 10],
            'min_samples_leaf': [1, 2, 3, 4, 5, 7, 9],
            'class_weight': [None, 'balanced']
        }

    elif model == 'rf':
        params_dict = {
            'n_estimators': [2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 110],
            'min_samples_split': [2, 3, 4, 5],
            'min_samples_leaf': [1, 2],
            'max_depth': [None, 25, 30, 35, 40, 45, 50],
            'class_weight': [None, 'balanced']
        }
    
    elif model == 'gbt':
        params_dict = {
            'learning_rate': [0.03, 0.05, 0.1],
            'n_estimators': [5, 10, 20, 30, 50, 70, 100],
            'max_depth': [None, 10, 20, 25, 30],
            'min_samples_split': [2, 3],
            'min_samples_leaf': [1, 3, 5, 7, 8],
            'class_weight': [None, 'balanced']
            }
    
    elif model == 'xgb':
        params_dict = {
            'n_estimators': [5, 10, 20, 30, 50, 100, 110, 120],
            'learning_rate': [0.05, 0.1],
            'min_child_weight': [1, 3],
            'max_depth': [None, 3, 6, 9],
            'gamma': [0, 0.001, 0.005, 0.01, 0.1, 1],
            'scale_pos_weight': [1, 3, 5, 10],
            'class_weight': [None]
        }
        
    elif model == 'lgb':
        params_dict = {
            'num_leaves': [11, 13, 21, 25, 31, 33, 43, 50, 70, 99],
            'max_depth': [-1, 5, 8, 15, 30],
            'n_estimators': [100, 110, 120],
            'min_child_samples': [10, 20, 25, 30],
            'class_weight': [None, 'balanced']
        }
    
    return params_dict


def get_474_params(model: str):
    if model == 'dt':
        params_dict = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 25, 30, 35, 40, 45, 50, 55],
            'min_samples_split': [2, 3, 4, 5, 7, 9, 10, 13],
            'min_samples_leaf': [1, 3, 5, 8],
            'class_weight': [None, 'balanced']
        }

    elif model == 'rf':
        params_dict = {
            'n_estimators': [3, 5, 10, 15, 30, 50, 70, 90, 100],
            'min_samples_split': [2, 3, 4, 5],
            'min_samples_leaf': [1, 2, 3],
            'max_depth': [None, 10, 25, 30, 35, 40, 50, 60],
            'class_weight': [None, 'balanced']
        }
    
    elif model == 'gbt':
        params_dict = {
            'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.3],
            'n_estimators': [10, 30, 50, 70, 90, 100, 130],
            'max_depth': [None, 3, 4, 7],
            'min_samples_split': [2, 3, 7, 10],
            'class_weight': [None, 'balanced']
            }
    
    elif model == 'xgb':
        params_dict = {
            'n_estimators': [10, 30, 35, 50, 100],
            'learning_rate': [0.03, 0.05, 0.1],
            'min_child_weight': [1, 3],
            'max_depth': [None, 3, 6, 9],
            'gamma': [0, 0.001, 0.005, 0.01, 0.1],
            'scale_pos_weight': [1, 3, 5, 7, 9, 10, 15],
            'class_weight': [None]
        }
        
    elif model == 'lgb':
        params_dict = {
            'num_leaves': [3, 5, 10, 15, 25, 33, 50, 70],
            'max_depth': [-1, 3, 5, 9, 15, 30],
            'n_estimators': [5, 35, 50, 70, 100, 150],
            'min_child_samples': [10, 20, 30, 35],
            'class_weight': [None, 'balanced']
        }
    
    return params_dict


def get_475_params(model: str):
    if model == 'dt':
        params_dict = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
            'min_samples_split': [2, 4, 5, 7, 9, 10, 13, 15, 20],
            'min_samples_leaf': [1, 3, 5, 7],
            'class_weight': [None, 'balanced']
        }

    elif model == 'rf':
        params_dict = {
            'n_estimators': [3, 5, 10, 15, 30, 50, 70, 90, 100],
            'min_samples_split': [2, 3, 4, 5],
            'min_samples_leaf': [1, 2, 3],
            'max_depth': [None, 10, 20, 30, 35, 40, 50],
            'class_weight': [None, 'balanced']
        }
    
    elif model == 'gbt':
        params_dict = {
            'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.3],
            'n_estimators': [50, 70, 90, 100, 110, 130],
            'max_depth': [None, 2, 4, 6],
            'min_samples_split': [2, 3, 4, 5],
            'class_weight': [None, 'balanced']
            }
    
    elif model == 'xgb':
        params_dict = {
            'n_estimators': [10, 30, 35, 50, 100],
            'learning_rate': [0.03, 0.05, 0.1],
            'min_child_weight': [1, 3],
            'max_depth': [None, 3, 6, 9],
            'gamma': [0, 0.001, 0.005, 0.01, 0.1],
            'scale_pos_weight': [1, 2, 3, 5, 7, 10],
            'class_weight': [None]
        }
        
    elif model == 'lgb':
        params_dict = {
            'num_leaves': [3, 5, 10, 15, 25, 33, 50, 70],
            'max_depth': [-1, 3, 5, 9, 15, 30],
            'n_estimators': [30, 50, 70, 100, 150],
            'min_child_samples': [5, 10, 20, 30, 35],
            'class_weight': [None, 'balanced']
        }
    
    return params_dict


def get_478_params(model: str):
    if model == 'dt':
        params_dict = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
            'min_samples_split': [2, 4, 5, 7, 9, 10, 13, 15, 20],
            'min_samples_leaf': [1, 3, 5, 7],
            'class_weight': [None, 'balanced']
        }

    elif model == 'rf':
        params_dict = {
            'n_estimators': [5, 10, 15, 30, 50, 70, 90, 100],
            'min_samples_split': [3, 5, 7, 10],
            'min_samples_leaf': [1, 2, 3],
            'max_depth': [None, 1, 3, 5, 7, 10, 20, 30],
            'class_weight': [None, 'balanced']
        }
    
    elif model == 'gbt':
        params_dict = {
            'learning_rate': [0.03, 0.05, 0.1, 0.3],
            'n_estimators': [30, 50, 70, 90, 100, 110],
            'max_depth': [None, 2, 7, 10],
            'min_samples_split': [2, 5, 9, 13],
            'class_weight': [None, 'balanced']
            }
    
    elif model == 'xgb':
        params_dict = {
            'n_estimators': [10, 25, 35, 50, 100],
            'learning_rate': [0.03, 0.05, 0.1],
            'min_child_weight': [1, 3],
            'max_depth': [None, 3, 6, 9],
            'gamma': [0, 0.001, 0.005, 0.01, 0.1],
            'scale_pos_weight': [1, 2, 3, 5, 7, 10],
            'class_weight': [None]
        }
        
    elif model == 'lgb':
        params_dict = {
            'num_leaves': [3, 5, 10, 15, 25, 33, 50, 70],
            'max_depth': [-1, 3, 5, 9, 15, 30],
            'n_estimators': [30, 50, 70, 100, 150],
            'min_child_samples': [5, 10, 20, 30, 35],
            'class_weight': [None, 'balanced']
        }
    
    return params_dict