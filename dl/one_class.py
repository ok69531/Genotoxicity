#%%
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

import random

from rdkit import Chem
from rdkit.Chem import MACCSkeys

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, 
    precision_score,
    recall_score,
    f1_score,
    accuracy_score, 
    precision_recall_curve,
    classification_report
)


#%%
tg_num = 471
path = f'../vitro/data/tg{tg_num}/tg{tg_num}.xlsx'

df = pd.read_excel(path)
mols = [Chem.MolFromSmiles(x) for x in df.SMILES]
x = np.array([list(MACCSkeys.GenMACCSKeys(x))[1:] for x in mols])
y = np.array([1 if x == 'positive' else 0 for x in df.Genotoxicity_maj])

x = torch.from_numpy(x)
y = torch.from_numpy(np.array(y))

np.unique(y, return_counts=True)
np.unique(y, return_counts=True)[1] / len(df)

positive_idx = torch.where(y==1)[0]
# positive_idx = torch.where(y==0)[0]

seed = 0
random.seed(seed)
train_idx = random.sample(range(len(positive_idx)), int(len(positive_idx) * 0.9))
test_idx = [i for i in range(len(df)) if i not in positive_idx[train_idx]]

train_x = x[positive_idx[train_idx]]
train_y = y[positive_idx[train_idx]]
test_x = x[test_idx]
test_y = y[test_idx]

val_x, test_x, val_y, test_y = train_test_split(test_x, test_y, test_size = 0.5, random_state = seed)


train_dataset = list(zip(train_x, train_y))
val_dataset = list(zip(val_x, val_y))
test_dataset = list(zip(test_x, test_y))


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
''' reconstruct할 때 0-1으로? 아니면 0~1사이 값으로 해서 BCE '''
class DeepSVDD(nn.Module):
    def __init__(self, input_dim, z_dim=32):
        super(DeepSVDD, self).__init__()
        
        self.lin1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.lin2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.lin3 = nn.Linear(256, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.lin4 = nn.Linear(64, z_dim)
        
    def forward(self, x):
        h = nn.functional.tanh(nn.functional.dropout(self.bn1(self.lin1(x)), 0.3))
        h = nn.functional.leaky_relu(self.bn2(self.lin2(h)))
        h = nn.functional.leaky_relu(self.bn3(self.lin3(h)))
        h = self.lin4(h)
        
        return h
        

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, z_dim = 32):
        super(AutoEncoder, self).__init__()
        
        self.lin1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.lin2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.lin3 = nn.Linear(256, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.lin4 = nn.Linear(64, z_dim)
        
        self.delin1 = nn.Linear(z_dim, 64)
        self.debn1 = nn.BatchNorm1d(64)
        self.delin2 = nn.Linear(64, 256)
        self.debn2 = nn.BatchNorm1d(256)
        self.delin3 = nn.Linear(256, 512)
        self.debn3 = nn.BatchNorm1d(512)
        self.delin4 = nn.Linear(512, input_dim)
        
    def encoder(self, x):
        h = nn.functional.tanh(nn.functional.dropout(self.bn1(self.lin1(x)), 0.3))
        h = nn.functional.leaky_relu(self.bn2(self.lin2(h)))
        h = nn.functional.leaky_relu(self.bn3(self.lin3(h)))
        h = self.lin4(h)
        
        return h
    
    def decoder(self, x):
        h = nn.functional.tanh(self.debn1(self.delin1(x)))
        h = nn.functional.leaky_relu(self.debn2(self.delin2(h)))
        h = nn.functional.leaky_relu(self.debn3(self.delin3(h)))
        h = nn.functional.sigmoid(self.delin4(h))
        
        return h
    
    def forward(self, x):
        z = self.encoder(x)
        output = self.decoder(z)
        
        return output


#%%
def pretrain(model, criterion, loader, device):
    model.train()
    
    total_loss = 0
    for (batch_x, _) in loader:
        batch_x = batch_x.to(torch.float32).to(device)
        
        pred = model(batch_x)
        
        optimizer.zero_grad()
        # reconst_loss = torch.mean(torch.sum((pred - batch_x)**2, dim = 1))
        reconst_loss = torch.mean(torch.sum(criterion(ae_model(batch_x), batch_x), dim = 1))
        total_loss += reconst_loss
        reconst_loss.backward()
        optimizer.step()
        
    scheduler.step()
    
    return total_loss/len(loader)


        

#%%
torch.manual_seed(seed)
torch.mps.manual_seed(seed)

lr = 0.005
input_dim = x.size(1)

ae_model = AutoEncoder(input_dim).to(device)
criterion = nn.BCEWithLogitsLoss(reduction = 'none')
# torch.sum(criterion(pred, batch_x), dim = 1)
optimizer = optim.Adam(ae_model.parameters(), lr = lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [50], gamma = 0.1)


best_train_loss = 1e+10
for i in range(1, 1000+1):
    loss = pretrain(ae_model, criterion, train_loader, device)
    
    if loss < best_train_loss:
        best_train_loss = loss
        best_epoch = i
        best_ae_param = ae_model.state_dict()
    
    if i % 10 == 0:
        print(f'epoch: {i}, train loss: {loss:.3f}')


torch.save(best_ae_param, 'autoencoder.pth')
ae_model.load_state_dict(best_ae_param)
ae_model.eval()


#%%
@torch.no_grad()
def get_center(model, loader, eps = 0.1):
    model.eval()
    
    latent = []
    for (batch_x, batch_y) in loader:
        batch_x = batch_x.to(torch.float32).to(device)
        h = model.encoder(batch_x)
        latent.append(h)
    
    latent = torch.cat(latent)
    center = torch.mean(latent, dim = 0)
    center[(abs(center) < eps) & (center < 0)] = -eps
    center[(abs(center) < eps) & (center > 0)] = eps
    
    return center


def finetune(model, loader, device, center):
    model.train()
    
    total_loss = 0
    for (batch_x, _) in loader:
        batch_x = batch_x.to(torch.float32).to(device)
        
        pred = model(batch_x)
        
        optimizer.zero_grad()
        reconst_loss = torch.mean(torch.sum((pred - center)**2, dim = 1))
        total_loss += reconst_loss
        reconst_loss.backward()
        optimizer.step()
        
    scheduler.step()
    
    return total_loss/len(loader)


@torch.no_grad()
def evaluate(model, loader, device, center):
    model.eval()
    
    scores, labels = [], []
    
    for (batch_x, batch_y) in loader:
        batch_x = batch_x.to(torch.float32).to(device)
        batch_y = batch_y.to(torch.float32)
        labels.append(batch_y)
        
        pred = model(batch_x)
        scores.append(torch.sum((pred - center)**2, dim = 1))
        
    scores = torch.cat(scores).detach().cpu().numpy()
    labels = torch.cat(labels).numpy()
    
    auc = roc_auc_score(labels, scores)
    
    return auc, scores, labels



#%%
torch.manual_seed(seed)
torch.mps.manual_seed(seed)

center = get_center(ae_model, train_loader)

model = DeepSVDD(input_dim).to(device)
model.load_state_dict(ae_model.state_dict(), strict=False)
optimizer = optim.Adam(ae_model.parameters(), lr = lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [50], gamma = 0.1)


best_val_loss = 1e+10
best_val_auc, best_test_auc, best_test_scores = 0, 0, 0
for i in range(1, 100+1):
    loss = finetune(model,train_loader, device, center)
    
    val_auc, val_scores, val_labels = evaluate(model, val_loader, device, center)
    test_auc, test_scores, test_labels = evaluate(model, test_loader, device, center)
    val_loss = val_scores.mean()
    test_loss = test_scores.mean()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        
        best_val_auc = val_auc
        best_test_auc = test_auc
        best_test_scores = test_scores
        
        best_epoch = i
        best_param = model.state_dict()
    
    # if i % 10 == 0:
    print(f'epoch: {i}')
    print(f'train loss: {loss:.3f}, val loss: {val_loss:.3f}, test loss: {test_loss:.3f}')
    print(f'val auc: {val_auc:.3f}, test auc: {test_auc:.3f}')


# %%
from sklearn.metrics import roc_curve, RocCurveDisplay

fpr, tpr, thresh = roc_curve(test_labels, best_test_scores)
display = RocCurveDisplay(fpr = fpr, tpr = tpr, roc_auc = best_test_auc)
display.plot()
test_labels


#%%
import matplotlib.pyplot as plt

# plt.hist(best_test_scores)

test_pred_label = [1 if x > 6 else 0 for x in best_test_scores]

print(classification_report(test_labels, test_pred_label))

plt.scatter(test_labels, best_test_scores)
# plt.boxplot(test_labels, best_test_scores)

# %%
