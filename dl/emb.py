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
    batch_x = batch_x.to(device)
    # batch_x = batch_x.to(torch.float32).to(device)
    batch_y = batch_y.to(torch.float32).to(device)
    break


#%%
class FingerprintsEncoder(nn.Module):
    def __init__(self, fp_length, emb_dim):
        super(FingerprintsEncoder, self).__init__()
        
        self.fp_embedding_list = nn.ModuleList()
        
        for i in range(fp_length):
            emb = nn.Embedding(emb_dim, emb_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            self.fp_embedding_list.append(emb)
        
    def forward(self, x):
        fp_embedding = []
        for i in range(x.shape[1]):
            fp_embedding.append(self.fp_embedding_list[i](x[:, i]))
        fp_embedding = torch.cat(fp_embedding, dim = 1)
        
        return fp_embedding


class MultiLayerPerceptron(nn.Module):
    def __init__(self, fp_length, emb_dim):
        super(MultiLayerPerceptron, self).__init__()
        
        self.fp_encoder = FingerprintsEncoder(fp_length = fp_length, emb_dim = emb_dim)
        
        self.lin1 = nn.Linear(fp_length*emb_dim, fp_length)
        self.lin2 = nn.Linear(fp_length, 64)
        self.lin3 = nn.Linear(64, 1)
    
    def forward(self, x):
        fp_embedding = self.fp_encoder(x)
        h = self.lin1(fp_embedding)
        h = nn.functional.leaky_relu(self.lin2(h))
        output = nn.functional.sigmoid(self.lin3(h))
        
        return output


class MLPwithLSTM(nn.Module):
    def __init__(self, fp_length, emb_dim):
        super(MLPwithLSTM, self).__init__()
        
        self.fp_encoder = FingerprintsEncoder(fp_length = fp_length, emb_dim = emb_dim)
        
        self.lstm = nn.LSTM(input_size = fp_length*emb_dim, hidden_size = 64, num_layers = 1)
        self.lin = nn.Linear(64, 1)
    
    def forward(self, x):
        fp_embedding = self.fp_encoder(x)
        h, _ = self.lstm(fp_embedding)
        output = nn.functional.sigmoid(self.lin(h))
        
        return output


class MLPwithTransformer(nn.Module):
    def __init__(self, fp_length, emb_dim):
        super(MLPwithTransformer, self).__init__()
        
        self.fp_encoder = FingerprintsEncoder(fp_length = fp_length, emb_dim = emb_dim)
        
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model = fp_length * emb_dim, nhead = emb_dim*2)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers = 6)
        
        self.lin1 = nn.Linear(fp_length * emb_dim, 64)
        self.lin2 = nn.Linear(64, 1)
        
        self.bn1 = nn.BatchNorm1d(fp_length*emb_dim)
    
    def forward(self, x):
        fp_embedding = self.fp_encoder(x)
        h = self.transformer_encoder(fp_embedding)
        h = nn.functional.gelu(self.bn1(h))
        h = nn.functional.gelu(self.lin1(h))
        output = nn.functional.sigmoid(self.lin2(h))
        
        return output    


#%%
class FLoss(nn.Module):
    def __init__(self, beta = 0.5, log_like = False):
        super(FLoss, self).__init__()
        
        self.beta = beta
        self.log_like = log_like
        
    def forward(self, pred, target):
        eps = 1e-10
        N = pred.size(0)
        TP = (pred * target).view(N, -1).sum()
        H = self.beta * target.view(N, -1).sum() + pred.view(N, -1).sum()
        fmeasure = (1 + self.beta) * TP / (H + eps)
        
        if self.log_like:
            floss = -torch.log(fmeasure)
        else:
            floss = (1 - fmeasure)
        
        return floss*100


class WeightedBCE(nn.Module):
    def __init__(self, weights):
        super(WeightedBCE, self).__init__()
        
        self.weights = weights
    
    def forward(self, pred, target):
        loss = - torch.mean(self.weights[1] * target * torch.log(pred) + self.weights[0] * (1 - target) * torch.log(1 - pred))

        return loss


#%%
def train(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    
    for (batch_x, batch_y) in loader:
        batch_x = batch_x.to(device)
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
        batch_x = batch_x.to(device)
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
emb_dim = 3
lr = 0.000005
# lr = 0.000003

val_aucs = []
val_f1s = []
test_aucs = []
test_f1s = []

# for seed in range(num_runs):
seed = 0
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

# weights = torch.tensor([5, 95])
# criterion = WeightedBCE(weights = weights)
# criterion = nn.BCELoss()
criterion = FLoss()

model = MLPwithTransformer(fp_length = x.shape[1], emb_dim = emb_dim).to(device)
# model = MLPwithLSTM(fp_length = x.shape[1], emb_dim = emb_dim).to(device)
# model = MultiLayerPerceptron(fp_length = x.shape[1], emb_dim = emb_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr = lr)
scheduler = optim.lr_scheduler.StepLR(optimizer = optimizer, step_size = 30, gamma = 0.1)

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
    
    # if epoch % 10 == 0:
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
print(f'test auc: {np.mean(test_aucs)*100:.2f}({np.std(test_aucs)*100:.2f})')
print(f'test f1: {np.mean(test_f1s)*100:.2f}({np.std(test_f1s)*100:.2f})')


#%%
model.eval()

test_y = []
test_pred_prob = []

for (batch_x, batch_y) in test_loader:
    test_y.append(batch_y)
    pred = model(batch_x.to(device))
    test_pred_prob.append(pred.view(-1).detach().cpu())


test_y = torch.cat(test_y).numpy()
test_pred_prob = torch.cat(test_pred_prob).numpy()

pred = np.where(test_pred_prob >= 0.8, 1, 0)

from sklearn.metrics import classification_report

print(classification_report(test_y, pred))
print(f'auc: {roc_auc_score(test_y, test_pred_prob)}')


# %%
a = FingerprintsEncoder(fp_length = x.size(1), emb_dim = 3).to(device)
el = nn.TransformerEncoderLayer(d_model = x.size(1)*3, nhead = 6).to(device)
te = nn.TransformerEncoder(el, num_layers = 6).to(device)

el(a(batch_x))
te(a(batch_x))