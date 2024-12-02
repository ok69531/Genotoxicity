# https://github.com/Samyu0304/graph-information-bottleneck-for-Subgraph-Recognition/tree/main

from multiprocessing.sharedctypes import Value
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
# from torch_geometric.nn import GINConv
from torch_geometric.utils import to_dense_adj

from .conv_layer import GINConv
from .encoder import AtomEncoder
from .floss import FLoss

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score
)

import warnings
warnings.filterwarnings('ignore')


class GIBGIN(nn.Module):
    def __init__(self, output_dim, num_layers, hidden, drop_ratio = 0.1):
        super(GIBGIN, self).__init__()
        
        self.drop_ratio = drop_ratio
        
        self.atom_encoder = AtomEncoder(emb_dim = hidden)
        
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GINConv(hidden))
        
        # subgraph generator layer (generate assignment matrix)
        self.cluster1 = Linear(hidden, hidden)
        self.cluster2 = Linear(hidden, 2)
        
        # classifier
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, output_dim)
        self.mse_loss = nn.MSELoss()
    
    def assignment(self, x):
        return self.cluster2(torch.tanh(self.cluster1(x)))
    
    def aggregate(self, assignment, x, batch, edge_index):
        
        max_id = torch.max(batch)
        EYE = torch.ones(2).to(edge_index.device)
        
        all_adj = to_dense_adj(edge_index, max_num_nodes = len(batch))[0]

        all_con_penalty = 0
        all_sub_embedding = []
        all_trivial_embedding = []
        all_graph_embedding = []
        active_node_index = []

        st = 0
        end = 0

        for i in range(int(max_id + 1)):
            j = 0
            while batch[st + j] == i and st + j <= len(batch) - 2:
                j += 1
            end = st + j

            if end == len(batch) - 1: 
                end += 1

            one_batch_x = x[st:end]
            one_batch_assignment = assignment[st:end]
            
            active = one_batch_assignment[:, 0]
            active = active > 0.5
            active = active.squeeze()
            active_nodes = active.nonzero().squeeze().tolist()
            
            subgraph_features = torch.mm(torch.t(one_batch_assignment), one_batch_x)
            trivial_features = subgraph_features[1].unsqueeze(dim = 0)
            subgraph_features = subgraph_features[0].unsqueeze(dim = 0) # S^T X: represetation of g_sub
                
            Adj = all_adj[st:end, st:end]
            new_adj = torch.mm(torch.t(one_batch_assignment), Adj)
            new_adj = torch.mm(new_adj, one_batch_assignment)
            normalize_new_adj = F.normalize(new_adj, p = 1, dim = 1)
            norm_diag = torch.diag(normalize_new_adj)
            con_penalty = self.mse_loss(norm_diag, EYE) # connectivity loss

            graph_embedding = torch.mean(one_batch_x, dim = 0, keepdim = True)

            all_sub_embedding.append(subgraph_features)
            all_trivial_embedding.append(trivial_features)
            all_graph_embedding.append(graph_embedding)
            active_node_index.append(active_nodes)

            all_con_penalty = all_con_penalty + con_penalty

            st = end

        all_sub_embedding = torch.cat(tuple(all_sub_embedding), dim = 0)
        all_trivial_embedding = torch.cat(tuple(all_trivial_embedding), dim = 0)
        all_graph_embedding = torch.cat(tuple(all_graph_embedding), dim = 0)
        all_con_penalty = all_con_penalty / (max_id + 1)
        
        return all_sub_embedding, all_trivial_embedding, all_graph_embedding, active_node_index, all_con_penalty
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        h = self.atom_encoder(x)
        # h = F.dropout(h, self.drop_ratio, training = self.training)
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)
        
        assignment = F.softmax(self.assignment(h), dim = 1)
        
        all_sub_embedding, all_trivial_embedding, all_graph_embedding, active_node_index, all_con_penalty = self.aggregate(assignment, h, batch, edge_index)
        
        h = F.relu(self.lin1(all_sub_embedding))
        h = F.dropout(h, p = 0.5, training = self.training)
        h = self.lin2(h)
        out = F.log_softmax(h, dim = -1)
        
        h_ = F.relu(self.lin1(all_trivial_embedding))
        h_ = F.dropout(h_, p = 0.5, training = False)
        h_ = self.lin2(h_)
        trivial_out = F.log_softmax(h_, dim = -1)
        
        return out, trivial_out, all_sub_embedding, all_graph_embedding, active_node_index, all_con_penalty
    
    def __repr__(self):
        return self.__class__.__name__


# for optimizing the graph and subgraph (phi_2)
class Discriminator(nn.Module):
    def __init__(self, hidden_size):
        super(Discriminator, self).__init__()
        
        self.input_size = 2 * hidden_size
        self.hidden_size = hidden_size
        
        self.lin1 = Linear(self.input_size, self.hidden_size)
        self.lin2 = Linear(self.hidden_size, 1)
        self.relu = ReLU()
        
    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        
    def forward(self, graph_embeddings, subgraph_embeddings):
        cat_embeddings = torch.cat((graph_embeddings, subgraph_embeddings), dim = -1)
        
        pre = self.relu(self.lin1(cat_embeddings))
        pre = self.relu(self.lin2(pre))
        
        return pre


def gib_train(model, discriminator, optimizer, local_optimizer, device, loader, gib_args, args):
    model.train()
    
    total_loss = 0
    
    for data in loader: 
        data = data.to(device)
        out, trivial_out, all_sub_embedding, all_graph_embedding, active_node_index, all_con_penalty = model(data)
        
        # to find phi_2^*
        for j in range(0, gib_args.inner_loop):
            local_optimizer.zero_grad()
            local_loss = -MI_Est(discriminator, all_graph_embedding.detach(), all_sub_embedding.detach())
            local_loss.backward()
            local_optimizer.step()
        
        optimizer.zero_grad()
        
        if args.target == 'maj':
            cls_loss = F.nll_loss(out, data.y_maj.view(-1), weight = torch.tensor([1., 10.]).to(device))
            # cls_loss = FLoss()(out[:, 1], data.y_maj.view(-1))
        elif args.target == 'consv':
            cls_loss = F.nll_loss(out, data.y_consv.view(-1), weight = torch.tensor([1., 10.]).to(device))
        mi_loss = MI_Est(discriminator, all_graph_embedding, all_sub_embedding)
        loss = (1 - gib_args.pp_weight) * (cls_loss + gib_args.beta * mi_loss) + gib_args.pp_weight * all_con_penalty
        
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    
    return total_loss / len(loader.dataset)


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def MI_Est(discriminator, graph_embeddings, sub_embeddings):
    batch_size = graph_embeddings.shape[0]
    shuffle_embeddings = graph_embeddings[torch.randperm(batch_size)]
    
    joint = discriminator(graph_embeddings, sub_embeddings)
    margin = discriminator(shuffle_embeddings, sub_embeddings)
    
    # Donsker
    mi_est = torch.mean(joint) - torch.clamp(torch.log(torch.mean(torch.exp(margin))), -100000, 100000)
    
    return mi_est


@torch.no_grad()
def gib_eval(model, device, loader, args):
    model.eval()
    
    loss = 0
    y = []
    sub_preds, trivial_preds = [], []
    sub_pred_prob, trivial_pred_prob = [], []
    for data in loader:
        data = data.to(device)
        
        out, trivial_out, _, _, _, _ = model(data)
        pred = out.max(1)[1]
        trivial_pred = trivial_out.max(1)[1]
        # correct += pred.eq(data.y.view(-1)).sum().item()
        
        if args.target == 'maj':
            y.append(data.y_maj)
        elif args.target == 'consv':
            y.append(data.y_consv)
        sub_preds.append(pred)
        trivial_preds.append(trivial_pred)
        sub_pred_prob.append(out[:, 1])
        trivial_pred_prob.append(trivial_out[:, 1])
        
        if args.target == 'maj':
            # loss_tmp = FLoss(beta=100)(out[:,1], data.y_maj.view(-1))
            # loss += loss_tmp.sum().item()
            loss += F.nll_loss(out, data.y_maj.view(-1), reduction='sum').item()
        elif args.target == 'consv':
            loss += F.nll_loss(out, data.y_consv.view(-1), reduction='sum').item()
    
    y = torch.cat(y).cpu().numpy()
    sub_preds = torch.cat(sub_preds).cpu().numpy()
    trivial_preds = torch.cat(trivial_preds).cpu().numpy()
    sub_pred_prob = torch.cat(sub_pred_prob).cpu().numpy()
    trivial_pred_prob = torch.cat(trivial_pred_prob).cpu().numpy()
    
    # sub_precision = precision_score(y, sub_preds, average = 'macro')
    subgraph_metric = {
        'accuracy': accuracy_score(y, sub_preds), 
        'precision': precision_score(y, sub_preds), 
        'recall': recall_score(y, sub_preds), 
        'f1': f1_score(y, sub_preds),
        'auc': roc_auc_score(y, sub_pred_prob)
    }
    
    trivial_metric = {
        'accuracy': accuracy_score(y, trivial_preds), 
        'precision': precision_score(y, trivial_preds), 
        'recall': recall_score(y, trivial_preds), 
        'f1': f1_score(y, trivial_preds),
        'auc': roc_auc_score(y, trivial_pred_prob)
    }
    
    return loss / len(loader.dataset), subgraph_metric, trivial_metric

