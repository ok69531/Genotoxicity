#%%
import os
import random
import warnings

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from tqdm import tqdm

from utils import *
from model import GNN, GNN_extract

from itertools import compress, repeat, product, chain
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, classification_report

warnings.filterwarnings('ignore')

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.data import Data, DataLoader, InMemoryDataset

print('Version of pytorch: ', torch.__version__)
print('Is GPU available?: ', torch.cuda.is_available())
print('GPU: ', torch.cuda.get_device_name(0))


#%%
device = torch.device("cuda:0")

dataset_name = "geno"

if dataset_name == "tox21":
    num_task = 12
elif dataset_name == "hiv":
    num_task = 1
elif dataset_name == "pcba":
    num_task = 128
elif dataset_name == "muv":
    num_task = 17
elif dataset_name == "bace":
    num_task = 1
elif dataset_name == "bbbp":
    num_task = 1
elif dataset_name == "toxcast":
    num_task = 617
elif dataset_name == "sider":
    num_task = 27
elif dataset_name == "clintox":
    num_task = 2
elif dataset_name == 'geno':
    num_task = 1



#%%
dataset = MoleculeDataset('dataset/' + dataset_name , dataset = dataset_name)
smiles_list = pd.read_csv('dataset/' + dataset_name + '/processed/smiles.csv', header=None)[0].tolist()

num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 


#%%
loader = DataLoader(dataset, batch_size = 32, shuffle = False, num_workers = 0)

model = GNN_extract(5, 300, num_tasks=num_task, JK='last', drop_ratio=0.2, graph_pooling='mean', gnn_type='gin').to(device)
model.from_pretrained('saved_model/init.pth')
model.eval()
model.to(device)

feature = []
y_true = []
for step, batch in enumerate(loader):
    batch = batch.to(device)
    
    with torch.no_grad():
        feature_ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

    feature.append(feature_)
    y_true.append(batch.y.view(len(feature_)))

feature = torch.cat(feature, dim = 0).cpu().numpy()
y_true = torch.cat(y_true, dim = 0).cpu().numpy()

np.unique(y_true)

y_true = [0 if i == -1 else 1 for i in y_true]


print('count', '\n', pd.Series(y_true).value_counts(),
      '\nratio', '\n',  pd.Series(y_true).value_counts(normalize = True).round(3))


#%%
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim, bias = True):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias = bias)
    
    def forward(self, x):
        output = torch.sigmoid(self.linear(x))
        return output


class RecallLoss(nn.Module):
    def __init__(self):
        super(RecallLoss, self).__init__()
        
    def forward(self, prob, y, recall, fpr):
        cond = (y_train == 1)
        loss = torch.mean(torch.where(cond, 
                                      -(1-recall) * torch.log(prob),
                                      -(1 - y_train) * fpr * torch.log(1 - prob)))
        return loss


# def train(model, optimizer):
#     model.train()
    
#     score = model(x_train)

#     optimizer.zero_grad()
#     loss = criterion(score.view(-1), y_train)
#     loss.backward()

#     optimizer.step()


def train(model, optimizer):
    model.train()
    
    score = model(x_train)
    pred = torch.where(score > 0.5, 1.0, 0.0)
    
    cm = confusion_matrix(y_train.cpu().data.numpy(), torch.squeeze(pred).cpu().data.numpy())
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    
    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)

    optimizer.zero_grad()
    loss = criterion(score, y_train, recall, fpr)
    loss.backward()

    optimizer.step()


# def l1_train(model, optimizer, c = 1.0, penalty = 'l1'):
#     model.train()
    
#     c = torch.FloatTensor([c]).to(device)
#     p = 1 if penalty == 'l1' else 2
    
#     score = model(x_train)
    
#     optimizer.zero_grad()
    
#     loss = criterion(score.view(-1), y_train)
#     regularity = torch.norm(model.linear.weight, p = p)
#     cost = loss + c * regularity
    
#     cost.backward()
#     optimizer.step()


# def l1_train(model, optimizer, c = 1.0, penalty = 'l1'):
#     model.train()
    
#     score = model(x_train)
#     pred = torch.where(score > 0.5, 1.0, 0.0)
    
#     cm = confusion_matrix(y_train.cpu().data.numpy(), torch.squeeze(pred).cpu().data.numpy())
#     tn = cm[0][0]
#     fp = cm[0][1]
#     fn = cm[1][0]
#     tp = cm[1][1]
    
#     recall = tp / (tp + fn)
#     fpr = fp / (fp + tn)

#     optimizer.zero_grad()
    
#     loss = criterion(score.view(-1), y_train, recall, fpr)
#     regularity = torch.norm(model.linear.weight, p = p)
#     cost = loss + c * regularity
    
#     cost.backward()
#     optimizer.step()


def eval(model):
    model.eval()
    
    with torch.no_grad():
        score = torch.squeeze(model(x_test))
        # loss = criterion(score, y_test)
        
    score = score.cpu().data.numpy()
    pred = np.where(score > 0.5, 1, 0)
    
    # return loss, score, pred
    return score, pred


#%%
test_roc = []
test_prec = []
test_recall = []
test_f1 = []
test_acc = []

for seed in tqdm(range(50)):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    feature = torch.Tensor(feature)
    y_true = torch.Tensor(y_true)

    train_idx = random.sample(range(len(feature)), int(len(feature)*0.8))
    test_idx = [i for i in range(len(feature)) if i not in train_idx]

    x_train = feature[torch.tensor(train_idx)].to(device)
    x_test = feature[torch.tensor(test_idx)].to(device)
    y_train = y_true[torch.tensor(train_idx)].to(device)
    y_test = y_true[torch.tensor(test_idx)].to(device)

    input_dim = feature.shape[1]
    output_dim = 1
    lr = 0.001
    
    model = LogisticRegression(input_dim, output_dim)
    model.to(device)
    
    criterion = RecallLoss()
    # criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr = lr)
    
    for epoch in range(300):
        train(model, optimizer)
        # l1_train(model, optimizer, c = 0.01, penalty = 'l1')
        
        test_score, test_pred = eval(model)
    
    y_test = y_test.cpu().data.numpy()
    test_roc.append(roc_auc_score(y_test, test_score))
    test_prec.append(precision_score(y_test, test_pred))
    test_recall.append(recall_score(y_test, test_pred))
    test_acc.append(accuracy_score(y_test, test_pred))    
    test_f1.append(f1_score(y_test, test_pred))


print('auc: ', np.mean(test_roc).round(3), '(', (np.std(test_roc, ddof = 1) / np.sqrt(len(test_roc))).round(3), ')',
      '\nprecision: ', np.mean(test_prec).round(3), '(', (np.std(test_prec, ddof = 1)/np.sqrt(len(test_prec))).round(3), ')',
      '\nrecall: ', np.mean(test_recall).round(3), '(', (np.std(test_recall, ddof = 1)/np.sqrt(len(test_recall))).round(3), ')',
      '\nacuracy: ', np.mean(test_acc).round(3), '(', (np.std(test_acc, ddof = 1)/np.sqrt(len(test_acc))).round(3) ,')',
      '\nf1: ', np.mean(test_f1).round(3), '(', (np.std(test_f1, ddof = 1)/np.sqrt(len(test_f1))).round(3), ')')

