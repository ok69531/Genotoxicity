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
        seed = [81, 152, 179, 409, 535, 604, 627, 961, 1067, 1185]
    elif tg_num == 474:
        seeds = [74, 248, 673, 1000, 1157, 1163, 1190, 1472, 1616, 1673] 
    elif tg_num == 475:
        seeds = [67, 74, 80, 97, 110, 152, 201, 213, 229, 238]
    elif tg_num == 478:
        seeds = [0, 4, 13, 16, 20, 30, 38, 41, 56, 59]
    
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
    
    return seeds


def save_model(file, path, name):
    if os.path.isdir(path):
        pass
    else:
        os.makedirs(path)
    torch.save(file, os.path.join(path, name))
    print('Parameters are successfully saved.')
