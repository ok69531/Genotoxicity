
#%%
import os
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy

from itertools import compress, repeat, product, chain
from collections import defaultdict

from sklearn.ensemble import RandomForestClassifier

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

import torch
import torch.nn.functional as F

from torch_geometric.data import Data, DataLoader, InMemoryDataset
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set

from torch_scatter import scatter_add

warnings.filterwarnings('ignore')


allowable_features = {
    'possible_atomic_num_list' : list(range(0, 119)),
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}


def mol_to_graph_data_obj_simple(smile, y):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    mol = Chem.MolFromSmiles(smile)
    y = torch.tensor(y).view(1, -1)
    
    # atoms
    num_atom_features = 2   # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
            'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2   # bond type, bond direction
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                                            'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(edge_index=edge_index, edge_attr=edge_attr, x=x, y = y, num_nodes = len(x))

    return data


#%%
num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 

class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        

    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, aggr = "add"):
        super(GINConv, self).__init__()
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GNN(torch.nn.Module):
    """
    

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr = "add"))
            # elif gnn_type == "gcn":
            #     self.gnns.append(GCNConv(emb_dim))
            # elif gnn_type == "gat":
            #     self.gnns.append(GATConv(emb_dim))
            # elif gnn_type == "graphsage":
            #     self.gnns.append(GraphSAGEConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        if x.dtype == torch.float32:
            x = x
        elif x.dtype == torch.int64:
            x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation


class GNN_extract(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """
    def __init__(self, num_layer, emb_dim, num_tasks, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin"):
        super(GNN_extract, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1
        
        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)

        return self.graph_pred_linear(self.pool(node_representation, batch))



#%%
def train(model, device, loader, criterion, optimizer):
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            optimizer.zero_grad()
            
            pred = model(batch)
            is_labeled = batch.y == batch.y
            loss = criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            
            loss.backward()
            optimizer.step()


def flag_train(model, device, loader, criterion, optimizer):
    model.train()
    
    m = 3
    emb_dim = 300
    max_pert = 0.001
    step_size = 0.001

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            optimizer.zero_grad()
            
            node_embedding = model.gnn.x_embedding1(batch.x[:, 0]) + model.gnn.x_embedding1(batch.x[:, 1])
            perturb = torch.FloatTensor(batch.num_nodes, emb_dim).uniform_(-max_pert, max_pert).to(device)
            perturb.requires_grad_()
            
            pred = model.graph_pred_linear(
                model.pool(
                    model.gnn(node_embedding + perturb, batch.edge_index, batch.edge_attr), 
                    batch.batch))
            y = batch.y
            is_labeled = y == y
            
            loss = criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss /= m
            
            for _ in range(m-1):
                loss.backward()
                perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
                perturb.data = perturb_data.data
                perturb.grad[:] = 0
                
                tmp_node_embedding = model.gnn.x_embedding1(batch.x[:, 0]) + model.gnn.x_embedding1(batch.x[:, 1]) + perturb
                tmp_pred = pred = model.graph_pred_linear(
                    model.pool(
                        model.gnn(tmp_node_embedding, batch.edge_index, batch.edge_attr), 
                        batch.batch))
                
                loss = 0
                loss = criterion(tmp_pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                loss /= m
                        
            loss.backward()
            optimizer.step()




@torch.no_grad()
def evaluation(model, device, loader, criterion):
    model.eval()
    
    y_true = []
    y_pred_prob = []
    loss_list = []
    
    for _, batch in enumerate(loader):
        batch = batch.to(device)
        
        if batch.x.shape[0] == 1:
            pass
        else:
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = torch.sigmoid(pred).to(torch.float32)
            y = batch.y.view(pred.shape).to(torch.float32)
            
            is_labeled = batch.y == batch.y
            
            y_true.append(y[is_labeled].detach().cpu())
            y_pred_prob.append(pred[is_labeled].detach().cpu())
            
            loss = criterion(pred[is_labeled], y[is_labeled])
            
            loss_list.append(loss)
        
    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred_prob = torch.cat(y_pred_prob, dim = 0).numpy()
    y_pred = np.where(y_pred_prob > 0.5, 1, 0)
    loss_list = torch.stack(loss_list)
    
    return sum(loss_list)/len(loss_list), roc_auc_score(y_true, y_pred_prob), precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred), accuracy_score(y_true, y_pred)



#%%
import sys
sys.path.append('../')

import os
import random
import numpy as np
from tqdm import tqdm

import torch
from torch import nn, optim

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from torch_geometric.utils import subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score, 
    accuracy_score
)


#%%
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)



#%%
tg_num = 471
path = f'../vitro/data/tg{tg_num}/tg{tg_num}.xlsx'

df = pd.read_excel(path)
y = df[['Genotoxicity_maj', 'Genotoxicity_consv']].to_numpy()
y = np.where(y == 'positive', 1, 0)


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


#%%
graph_list = [mol_to_graph_data_obj_simple(df.SMILES[i], y[i]) for i in range(len(df))]

train_dataset = [graph_list[i] for i in train_idx]
valid_dataset = [graph_list[i] for i in valid_idx]
test_dataset = [graph_list[i] for i in test_idx]

train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
valid_loader = DataLoader(valid_dataset, batch_size = 32, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)


for _, batch in enumerate(train_loader):
    break


#%%
num_runs = 3
num_task = 2
num_layer = 5
emb_dim = 300
gnn_type = 'gin'
graph_pooling = 'mean'
dropout_ratio = 0.3
JK = 'last'

epochs = 100

lr = 0.001
optim_method = 'adam'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
criterion = nn.BCEWithLogitsLoss()


auc_vals, auc_tests = [], []
prec_vals, prec_tests = [], []
rec_vals, rec_tests = [], []
f1_vals, f1_tests = [], []
acc_vals, acc_tests = [], []

for seed in range(num_runs):
    print(f'====================== run: {seed} ======================')
    
    set_seed(seed)
    torch_geometric.seed_everything(seed)
    
    best_val_auc, final_test_auc = 0, 0
    best_val_prec, final_test_prec = 0, 0
    best_val_rec, final_test_rec = 0, 0
    best_val_f1, final_test_f1 = 0, 0
    best_val_acc, final_test_acc = 0, 0
    best_val_loss, final_test_loss = 100, 100

    model = GNN_extract(num_layer = num_layer, emb_dim = emb_dim, num_tasks = num_task, 
                        JK = 'last', drop_ratio = dropout_ratio, graph_pooling = graph_pooling, 
                        gnn_type = gnn_type)
    model.from_pretrained('mgssl.pth')
    model = model.to(device)
    
    if optim_method == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr = lr)
    elif optim_method == 'adam':
        optimizer = optim.Adam(model.parameters(), lr = lr)
    
    for epoch in range(1, epochs + 1):
        print(f'=== epoch {epoch}')
        
        # train(model, device, train_loader, criterion, optimizer)
        # flag_train(model, device, train_loader, criterion, optimizer)
        
        train_loss, train_auc, train_prec, train_rec, train_f1, train_acc = evaluation(model, device, train_loader, criterion)
        val_loss, val_auc, val_prec, val_rec, val_f1, val_acc = evaluation(model, device, valid_loader, criterion)
        test_loss, test_auc, test_prec, test_rec, test_f1, test_acc = evaluation(model, device, test_loader, criterion)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc
            best_val_prec = val_prec
            final_test_prec = test_prec
            best_val_rec = val_rec
            final_test_rec = test_rec
            best_val_f1 = val_f1
            final_test_f1 = test_f1
            best_val_acc = val_acc
            final_test_acc = test_acc
            
            best_epoch = epoch
            model_params = model.state_dict()

        print(f'train loss: {train_loss:.4f}, train auc: {train_auc*100:.2f}')
        print(f'val loss: {val_loss:.4f}, val auc: {val_auc*100:.2f}')
        print(f'test loss: {test_loss:.4f}, test auc: {test_auc*100:.2f}')
        
    auc_vals.append(best_val_auc)
    auc_tests.append(final_test_auc)
    prec_vals.append(best_val_prec)
    prec_tests.append(final_test_prec)
    rec_vals.append(best_val_rec)
    rec_tests.append(final_test_rec)
    f1_vals.append(best_val_f1)
    f1_tests.append(final_test_f1)
    acc_vals.append(best_val_acc)
    acc_tests.append(final_test_acc)
        
print('')
print(f'Validation auc: {np.mean(auc_vals)*100:.2f}({np.std(auc_vals)*100:.2f})')
print(f'Test auc: {np.mean(auc_tests)*100:.2f}({np.std(auc_tests)*100:.2f})')
print(f'Validation precision: {np.mean(prec_vals)*100:.2f}({np.std(prec_vals)*100:.2f})')
print(f'Test precision: {np.mean(prec_tests)*100:.2f}({np.std(prec_tests)*100:.2f})')
print(f'Validation recall: {np.mean(rec_vals)*100:.2f}({np.std(rec_vals)*100:.2f})')
print(f'Test recall: {np.mean(rec_tests)*100:.2f}({np.std(rec_tests)*100:.2f})')
print(f'Validation f1-score: {np.mean(f1_vals)*100:.2f}({np.std(f1_vals)*100:.2f})')
print(f'Test f1-score: {np.mean(f1_tests)*100:.2f}({np.std(f1_tests)*100:.2f})')
print(f'Validation accuracy: {np.mean(acc_vals)*100:.2f}({np.std(acc_vals)*100:.2f})')
print(f'Test accuracy: {np.mean(acc_tests)*100:.2f}({np.std(acc_tests)*100:.2f})')


#%%

