#%%
import numpy as np
import pandas as pd

import rdkit
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

from imblearn.over_sampling import SMOTE, SVMSMOTE

import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300


#%%
tg_num = 471
path = f'vitro/data/tg{tg_num}/tg{tg_num}.xlsx'

df = pd.read_excel(path)
df.Genotoxicity_maj.value_counts()
df.Genotoxicity_maj.value_counts(normalize = True)
df.Genotoxicity_consv.value_counts()
df.Genotoxicity_consv.value_counts(normalize = True)


mols = [Chem.MolFromSmiles(x) for x in df.SMILES]

ecfp = np.array([list(AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=1024)) for x in mols])
maccs = np.array([list(MACCSkeys.GenMACCSKeys(x)) for x in mols])

maj_y = [1 if x == 'positive' else 0 for x in df.Genotoxicity_maj]
consv_y = [1 if x == 'positive' else 0 for x in df.Genotoxicity_consv]


#%%
seed = 0

train_ecfp, test_ecfp, train_maj_y, test_maj_y = train_test_split(ecfp, maj_y, test_size = 0.2, random_state = seed)
train_ecfp, test_ecfp, train_consv_y, test_consv_y = train_test_split(ecfp, consv_y, test_size = 0.2, random_state = seed)
train_maccs, test_maccs, train_maj_y, test_maj_y = train_test_split(maccs, maj_y, test_size = 0.2, random_state = seed)
train_maccs, test_maccs, train_consv_y, test_consv_y = train_test_split(maccs, consv_y, test_size = 0.2, random_state = seed)

model = RandomForestClassifier(random_state = seed)
model.fit(train_maccs, train_maj_y)
pred = model.predict(test_maccs)
pred_prob = model.predict_proba(test_maccs)[:, 1]

print(classification_report(test_maj_y, pred))
print(f'auc: {roc_auc_score(test_maj_y, pred_prob)}')

# model.feature_importances_

# plt.hist(model.feature_importances_)
# plt.show()
# plt.close()


#%%
sm = SVMSMOTE(random_state = seed)

ov_train_maccs, ov_train_maj_y = sm.fit_resample(train_maccs, train_maj_y)

model = RandomForestClassifier(random_state = seed)
model.fit(ov_train_maccs, ov_train_maj_y)
pred = model.predict(test_maccs)
pred_prob = model.predict_proba(test_maccs)[:, 1]

print(classification_report(test_maj_y, pred))
print(f'auc: {roc_auc_score(test_maj_y, pred_prob)}')

model.feature_importances_

# plt.hist(model.feature_importances_)
# plt.show()
# plt.close()


#%%
# https://github.com/optuna/optuna-examples/tree/main
# https://www.kaggle.com/code/muhammetgamal5/kfold-cross-validation-optuna-tuning


import optuna
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate, StratifiedKFold

def binary_cross_validation(model, x, y, seed, smote = False):
    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
    
    metrics = ['f1', 'auc']
    
    train_metrics = list(map(lambda x: 'train_' + x, metrics))
    val_metrics = list(map(lambda x: 'val_' + x, metrics))
    
    train_f1 = []
    train_auc = []
    
    val_f1 = []
    val_auc = []
    
    for train_idx, val_idx in skf.split(x, y):
        train_x, train_y = x[train_idx], y[train_idx]
        val_x, val_y = x[val_idx], y[val_idx]
        
        if smote:
            sm = SVMSMOTE(random_state = seed)
            train_x, train_y = sm.fit_resample(train_x, train_y)
        else:
            pass
        
        model.fit(train_x, train_y)
        
        train_pred = model.predict(train_x)
        train_pred_score = model.predict_proba(train_x)[:, 1]
        
        val_pred = model.predict(val_x)
        val_pred_score = model.predict_proba(val_x)[:, 1]
        
        train_f1.append(f1_score(train_y, train_pred))
        train_auc.append(roc_auc_score(train_y, train_pred_score))

        val_f1.append(f1_score(val_y, val_pred))
        val_auc.append(roc_auc_score(val_y, val_pred_score))

    result = dict(zip(train_metrics + val_metrics, 
                      [np.mean(train_f1), np.mean(train_auc), np.mean(val_f1), np.mean(val_auc)]))
    
    return result


def objective(trial):
    params = {
        'random_state': trial.suggest_int('random_state', 0, 0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 150),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
    }
    
    clf = RandomForestClassifier(**params)
    
    # cv_result = cross_validate(clf, train_maccs, train_maj_y, scoring = 'f1_macro')
    # f1 = cv_result['test_score'].mean()
    
    cv_result = binary_cross_validation(clf, train_maccs, np.array(train_maj_y), seed = 0, smote = True)
    f1 = cv_result['val_f1']
    
    return f1
    
study = optuna.create_study(direction = 'maximize')
study.optimize(objective, n_trials = 100)
len(study.trials)
ㄱstudy.best_trial.params

smote = True
if smote:
    sm = SVMSMOTE(random_state=0)
    ov_x, ov_y = sm.fit_resample(train_maccs, train_maj_y)
else:
    pass

clf = RandomForestClassifier(**study.best_trial.params)
if smote:
    clf.fit(ov_x, ov_y)
else:
    clf.fit(train_maccs, train_maj_y)
pred = clf.predict(test_maccs)
pred_prob = clf.predict_proba(test_maccs)[:, 1]

print(classification_report(test_maj_y, pred))
print(f'auc: {roc_auc_score(test_maj_y, pred_prob)}')


#%%
import random

import torch
from torch import nn, optim
from torch.utils.data import DataLoader


#%%
seed = 0
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

maccs = torch.from_numpy(maccs)
maj_y = torch.from_numpy(np.array(maj_y))

train_dataset = list(zip(maccs[train_idx], maj_y[train_idx]))
val_dataset = list(zip(maccs[valid_idx], maj_y[valid_idx]))
test_dataset = list(zip(maccs[test_idx], maj_y[test_idx]))


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
    def __init__(self, input_size):
        super(MultiLayerPerceptron, self).__init__()
        
        # self.lin1 = nn.Linear(input_size, 1024)
        # self.lin2 = nn.Linear(1024, 512)
        # self.lin3 = nn.Linear(512, 256)
        self.lin4 = nn.Linear(input_size, 64)
        self.lin5 = nn.Linear(64, 1)
        
        # self.bn1 = nn.BatchNorm1d(1024)
        # self.bn2 = nn.BatchNorm1d(512)
        # self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        # h = self.bn1(self.lin1(x))
        # h = nn.functional.leaky_relu(h)
        # h = self.bn2(self.lin2(h))
        # h = nn.functional.leaky_relu(h)
        # h = self.bn3(self.lin3(h))
        # h = nn.functional.leaky_relu(h)
        h = self.bn4(self.lin4(x))
        h = nn.functional.relu(h)
        # h = nn.functional.leaky_relu(h)
        h = self.lin5(h)
        
        return h


def train(model, optimizer, criterion, loader, device):
    model.train()
    
    for (batch_x, batch_y) in loader:
        batch_x = batch_x.to(torch.float32).to(device)
        batch_y = batch_y.to(torch.float32).to(device)
        
        optimizer.zero_grad()
        
        pred = model(batch_x)
        loss = criterion(pred, batch_y.view(pred.shape))
        
        loss.backward()
        optimizer.step()


def aa_train(model, optimizer, criterion, loader, device):
    model.train()
    
    for (batch_x, batch_y) in loader:
        batch_x = batch_x.to(torch.float32).to(device)
        batch_y = batch_y.to(torch.float32).to(device)
        
        optimizer.zero_grad()
        
        perturb = torch.FloatTensor(batch_x.size()).uniform_(-0.01, 0.01).to(device)
        perturb.requires_grad_()
        
        pred = model(batch_x + perturb)
        loss = criterion(pred, batch_y.view(pred.shape))
        loss /= 3
        
        for _ in range(2):
            loss.backward()
            
            perturb_data = perturb.detach() + 0.001 * torch.sign(perturb.grad.detach())
            perturb.data = perturb_data.data
            perturb.grad[:] = 0
            
            pred = model(batch_x + perturb)
            
            loss = 0
            loss = criterion(pred, batch_y.view(pred.shape))
            loss /= 3
        
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(model, criterion, loader, device):
    model.eval()
    
    loss = 0
    y_true = []
    y_pred = []
    
    for (batch_x, batch_y) in loader:
        batch_x = batch_x.to(torch.float32).to(device)
        batch_y = batch_y.to(torch.float32).to(device)
        
        pred = model(batch_x)
        loss += criterion(pred, batch_y.view(pred.shape))

        y_true.append(batch_y.cpu())
        y_pred.append(pred.view(-1).cpu())
    
    loss = loss / len(loader)
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    
    return loss, y_true, y_pred


#%%
seed = 0
torch.manual_seed(seed)
torch.mps.manual_seed(seed)

input_size = maccs.size(1)
model = MultiLayerPerceptron(input_size).to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.BCEWithLogitsLoss()


best_val_auc, best_test_auc = 0, 0
for epoch in range(1, 100 + 1):
    # aa_train(model, optimizer, criterion, train_loader, device)
    train(model, optimizer, criterion, train_loader, device)
    
    train_loss, train_y_true, train_y_pred = evaluate(model, criterion, train_loader, device)
    val_loss, val_y_true, val_y_pred = evaluate(model, criterion, val_loader, device)
    test_loss, test_y_true, test_y_pred = evaluate(model, criterion, test_loader, device)
    
    train_auc = roc_auc_score(train_y_true, train_y_pred)
    val_auc = roc_auc_score(val_y_true, val_y_pred)
    test_auc = roc_auc_score(test_y_true, test_y_pred)
    
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_test_auc = test_auc
        
        model_params = model.state_dict()
    
    if epoch % 10 == 0:
        print(f'=== epoch {epoch} ===')
        print(f'train loss: {train_loss:.5f}, validation loss: {val_loss:.5f}, test loss: {test_loss:.5f}')
        print(f'train auc: {train_auc:.3f}, validation auc: {val_auc:.3f}, test auc: {test_auc:.3f}')


#%%
model.load_state_dict(model_params)
model.eval()

_, test_y_true, test_y_pred = evaluate(model, criterion, test_loader, device)
test_y_pred_prob = nn.functional.sigmoid(test_y_pred)
pred = torch.tensor([1. if x>= 0.5 else 0. for x in test_y_pred_prob])

print(classification_report(test_y_true, pred))
print(f'auc: {roc_auc_score(test_y_true, pred):.3f}')


# %%
