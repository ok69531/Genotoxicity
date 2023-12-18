#%%
import random

import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import rdkit
from rdkit import Chem
from rdkit.Chem import MACCSkeys

from sklearn.metrics import roc_auc_score, f1_score


#%%
tg_num = 471
path = f'../vitro/data/tg{tg_num}/tg{tg_num}.xlsx'

df = pd.read_excel(path)
mols = [Chem.MolFromSmiles(x) for x in df.SMILES]
x = np.array([list(MACCSkeys.GenMACCSKeys(x))[1:] for x in mols])
y = np.array([1 if x == 'positive' else 0 for x in df.Genotoxicity_maj])

x = torch.from_numpy(x)
y = torch.from_numpy(np.array(y))


#%%
seed = 42
random.seed(seed)

num_mols = len(df)
all_idx = list(range(num_mols))
random.shuffle(all_idx)

frac_train = 0.8
frac_valid = 0.1

train_idx = all_idx[: int(frac_train * num_mols)]
valid_idx = all_idx[int(frac_train * num_mols) : int(frac_valid * num_mols) + int(frac_train * num_mols)]
test_idx = all_idx[int(frac_valid * num_mols) + int(frac_train * num_mols):]

assert len(train_idx) + len(valid_idx) + len(test_idx) == len(df)

train_dataset = list(zip(x[train_idx], y[train_idx]))
val_dataset = list(zip(x[valid_idx], y[valid_idx]))
test_dataset = list(zip(x[test_idx], y[test_idx]))


#%%
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

for (batch_x, batch_y) in train_loader:
    batch_x = batch_x.to(torch.float32).to(device)
    batch_y = batch_y.to(torch.float32).to(device)
    break


#%%
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim):
        super(MultiLayerPerceptron, self).__init__()
        
        self.lin1 = nn.Linear(input_dim, 2*input_dim)
        self.lin2 = nn.Linear(2*input_dim, input_dim)
        self.lin3 = nn.Linear(input_dim, 64)
        self.lin4 = nn.Linear(64, 1)
        
    def forward(self, x):
        h = nn.functional.leaky_relu(self.lin1(x))
        h = nn.functional.leaky_relu(self.lin2(h))
        h = nn.functional.leaky_relu(self.lin3(h))
        output = nn.functional.sigmoid(self.lin4(h))
        
        return output


#%%
def train(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    
    for (batch_x, batch_y) in loader:
        batch_x = batch_x.to(torch.float32).to(device)
        batch_y = batch_y.to(torch.float32).to(device)
        
        optimizer.zero_grad()
        
        pred = model(batch_x)
        loss = criterion(pred, batch_y.view(pred.shape))
        
        loss.backward()
        optimizer.step()
    scheduler.step()


@torch.no_grad()
def evaluation(model, loader, criterion, device):
    model.eval()
    
    y_true = []
    y_pred_prob = []
    loss = []
    
    for (batch_x, batch_y) in loader:
        batch_x = batch_x.to(torch.float32).to(device)
        batch_y = batch_y.to(torch.float32).to(device)
        
        pred = model(batch_x)
        loss.append(criterion(pred, batch_y.view(pred.shape)))
        
        y_pred_prob.append(pred.view(-1).detach().cpu())
        y_true.append(batch_y.view(-1).cpu())
        
    y_true = torch.cat(y_true).numpy()
    y_pred_prob = torch.cat(y_pred_prob).numpy()
    y_pred = np.where(y_pred_prob >= 0.5, 1, 0)
    
    loss = torch.stack(loss)
    auc = roc_auc_score(y_true, y_pred_prob)
    f1 = f1_score(y_true, y_pred)
    
    return (sum(loss)/len(loss)).item(), auc, f1


#%%
num_runs = 10

val_aucs = []
val_f1s = []
test_aucs = []
test_f1s = []

for seed in range(num_runs):
    print(f'========== run {seed} ==========')
    random.seed(seed)

    num_mols = len(df)
    all_idx = list(range(num_mols))
    random.shuffle(all_idx)

    frac_train = 0.8
    frac_valid = 0.1

    train_idx = all_idx[: int(frac_train * num_mols)]
    valid_idx = all_idx[int(frac_train * num_mols) : int(frac_valid * num_mols) + int(frac_train * num_mols)]
    test_idx = all_idx[int(frac_valid * num_mols) + int(frac_train * num_mols):]

    assert len(train_idx) + len(valid_idx) + len(test_idx) == len(df)

    train_dataset = list(zip(x[train_idx], y[train_idx]))
    val_dataset = list(zip(x[valid_idx], y[valid_idx]))
    test_dataset = list(zip(x[test_idx], y[test_idx]))
    
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

    criterion = nn.BCELoss()

    model = MultiLayerPerceptron(x.size(1)).to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer = optimizer, lr_lambda = lambda epoch: 0.95**epoch, last_epoch = -1, verbose = False)

    epochs = 100

    best_val_f1, final_test_f1 = 0, 0
    for epoch in range(1, epochs + 1):
        train(model, train_loader, criterion, optimizer, scheduler, device)
        
        train_loss, train_auc, train_f1 = evaluation(model, train_loader, criterion, device)
        val_loss, val_auc, val_f1 = evaluation(model, val_loader, criterion, device)
        test_loss, test_auc, test_f1 = evaluation(model, test_loader, criterion, device)
        
        if val_f1 > best_val_f1:
            model_params = model.state_dict()
            best_val_f1 = val_f1
            best_val_auc = val_auc
            final_test_f1 = test_f1
            final_test_auc = test_auc
        
        if epoch % 10 == 0:
            print(f'=== epoch: {epoch} ===')
            print(f'train loss: {train_loss:.3f}, validation loss: {val_loss:.3f}, test loss: {test_loss:.3f}')
            print(f'train auc: {train_auc:.3f}, validation auc: {val_auc:.3f}, test auc: {test_auc:.3f}')
            print(f'train f1: {train_f1:.3f}, validation f1: {val_f1:.3f}, test f1: {test_f1:.3f}')

    val_aucs.append(best_val_auc)
    val_f1s.append(best_val_f1)
    test_aucs.append(final_test_auc)
    test_f1s.append(final_test_f1)

print('')
print(f'val auc: {np.mean(val_aucs)*100:.2f}({np.std(val_aucs)*100:.2f})')
print(f'val f1: {np.mean(val_f1s)*100:.2f}({np.std(val_f1s)*100:.2f})')
print(f'val auc: {np.mean(test_aucs)*100:.2f}({np.std(test_aucs)*100:.2f})')
print(f'val f1: {np.mean(test_f1s)*100:.2f}({np.std(test_f1s)*100:.2f})')
    

#%%