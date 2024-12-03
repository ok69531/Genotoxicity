import os
import random
import numpy as  np

import torch
import torch_geometric


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch_geometric.seed_everything(seed)


def get_seed(tg_num):
    if tg_num == 471:
        seeds = [4, 47, 61, 67, 73, 76, 79, 88, 106, 123]
    elif tg_num == 473:
        seeds = [6, 31, 37, 76, 158, 288, 314, 347, 380, 396]
    elif tg_num == 476:
        seeds = [342, 435, 870, 956, 973, 1096, 1181, 1188, 1312, 1394]
    elif tg_num == 487:
        seeds = [81, 152, 179, 409, 535, 604, 627, 961, 1067, 1185]
    elif tg_num == 474:
        seeds = [74, 248, 673, 1000, 1157, 1163, 1190, 1472, 1616, 1673] 
    elif tg_num == 475:
        seeds = [74, 97, 152, 213, 229, 312, 341, 353, 360, 371]
    elif tg_num == 478:
        seeds = [0, 13, 30, 38, 41, 56, 59, 63, 66, 74]
    
    return seeds


def save_model(file, path, name):
    if os.path.isdir(path):
        pass
    else:
        os.makedirs(path)
    torch.save(file, os.path.join(path, name))
    print('Parameters are successfully saved.')
