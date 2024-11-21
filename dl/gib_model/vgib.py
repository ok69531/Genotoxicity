import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn import GINConv
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn.glob import global_mean_pool, global_add_pool, global_max_pool

from .conv_layer import GINConv
from .encoder import AtomEncoder

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score
)

import warnings
warnings.filterwarnings('ignore')


class VariationalGIB(nn.Module):
    def __init__(self, args, drop_ratio = 0.1):
        super(VariationalGIB, self).__init__()
        
        self.args = args
        self.hidden = args.hidden
        self.drop_ratio = drop_ratio
        
        self.mseloss = nn.MSELoss()
        self.relu = nn.ReLU()
        
        self.atom_encoder = AtomEncoder(emb_dim = self.hidden)
        
        # self.graph_convolution_1 = GINConv(self.hidden)
        # self.graph_convolution_2 = GINConv(self.hidden)
        
        self.convs = nn.ModuleList()
        for i in range(self.args.num_layers):
            self.convs.append(GINConv(self.args.hidden))

        self.fully_connected_1 = nn.Linear(self.hidden, self.hidden)
        self.fully_connected_2 = nn.Linear(self.hidden, self.args.second_dense_neurons)

    def gumbel_softmax(self, prob):
        return F.gumbel_softmax(prob, tau = 1, dim = -1)

    def forward(self, data):
        epsilon = 0.0000001
        
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        h = self.atom_encoder(x)
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)

        node_feature = h
        all_adj = to_dense_adj(edge_index, max_num_nodes = len(batch))[0]

        all_kl_loss = 0
        all_pos_penalty = 0
        all_preserve_rate = 0
        all_graph_embedding = []
        all_noisy_embedding = []
        active_node_index = []

        st = 0
        end = 0
        max_id = torch.max(batch)

        for i in range(int(max_id + 1)):
            j = 0
            while batch[st + j] == i and st + j <= len(batch) - 2:
                j += 1
            end = st + j

            if end == len(batch) - 1:
                end += 1
            
            one_batch_x = node_feature[st:end]
            num_nodes = one_batch_x.size(0)
            
            # this part is used to add noise
            static_node_feature = one_batch_x.clone().detach()
            node_feature_std, node_feature_mean = torch.std_mean(static_node_feature, dim = 0)

            # this part is used to generate assignment matrix
            abstract_features_1 = torch.tanh(self.fully_connected_1(one_batch_x))
            assignment = F.softmax(self.fully_connected_2(abstract_features_1), dim = 1)
            gumbel_assignment = self.gumbel_softmax(assignment)

            # graph embedding
            graph_feature = torch.sum(one_batch_x, dim = 0, keepdim = True)

            # add noise to the node representation
            node_feature_mean = node_feature_mean.repeat(num_nodes, 1)

            # noisy graph representation
            lambda_pos = gumbel_assignment[:, 0].unsqueeze(dim = 1)
            lambda_neg = gumbel_assignment[:, 1].unsqueeze(dim = 1)
            
            active = lambda_pos > 0.5
            active = active.squeeze()
            active_nodes = active.nonzero().squeeze().tolist()

            noisy_node_feature_mean = lambda_pos * one_batch_x + lambda_neg * node_feature_mean
            noisy_node_feature_std = lambda_neg * node_feature_std

            noisy_node_feature = noisy_node_feature_mean + torch.rand_like(noisy_node_feature_mean) * noisy_node_feature_std
            noisy_graph_feature = torch.sum(noisy_node_feature, dim = 0, keepdim = True)

            KL_tensor = 0.5 * ((noisy_node_feature_std ** 2) / (node_feature_std+epsilon) ** 2) + \
                torch.sum(((noisy_node_feature_mean -node_feature_mean)/(node_feature_std + epsilon))**2, dim = 0)
            KL_loss = torch.mean(KL_tensor)

            # if torch.cuda.is_available():
            #     EYE = torch.ones(2).cuda()
            #     Pos_mask = torch.FloatTensor([1, 0]).cuda()
            # else:
            EYE = torch.ones(2).to(edge_index.device)
            Pos_mask = torch.FloatTensor([1, 0]).to(edge_index.device)

            Adj = all_adj[st:end,st:end]
            Adj.requires_grad = False
            new_adj = torch.mm(torch.t(assignment), Adj)
            new_adj = torch.mm(new_adj, assignment)

            normalize_new_adj = F.normalize(new_adj, p = 1, dim = 1)
            norm_diag = torch.diag(normalize_new_adj)
            pos_penalty = self.mseloss(norm_diag, EYE)
            
            # cal preserve rate (?)
            preserve_rate = torch.sum(assignment[:, 0] > 0.5) / assignment.size(0)

            all_kl_loss = all_kl_loss + KL_loss
            all_pos_penalty = all_pos_penalty + pos_penalty
            all_preserve_rate = all_preserve_rate + preserve_rate

            all_graph_embedding.append(graph_feature)
            all_noisy_embedding.append(noisy_graph_feature)
            active_node_index.append(active_nodes)
            
            st = end

        all_graph_embedding = torch.cat(tuple(all_graph_embedding), dim = 0)
        all_noisy_embedding = torch.cat(tuple(all_noisy_embedding), dim = 0)
        all_pos_penalty = all_pos_penalty / (max_id + 1)
        all_kl_loss = all_kl_loss / (max_id + 1)
        all_preserve_rate = all_preserve_rate / (max_id + 1)
        
        return all_graph_embedding, all_noisy_embedding, active_node_index, all_pos_penalty, all_kl_loss, all_preserve_rate


class Classifier(nn.Module):
    def __init__(self, args, num_class):
        super(Classifier, self).__init__()
        
        self.args = args
        self.lin1 = nn.Linear(self.args.hidden, self.args.cls_hidden_dimensions)
        self.lin2 = nn.Linear(self.args.cls_hidden_dimensions, num_class)
        self.relu = nn.ReLU()
    
    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        
    def forward(self, data):
        out = self.lin1(data)
        out = self.relu(out)
        out = self.lin2(out)
        
        return F.log_softmax(out, dim = -1)


def vgib_train(model, classifier, optimizer, device, loader, args):
    model.train()
    
    total_loss = 0
    
    for data in loader: 
        data = data.to(device)
        num_graphs = len(data.y)
        
        embedding, noisy, active_node_index, pos_penalty, kl_loss, preserve_rate = model(data)
        features = noisy
        labels = data.y
        
        # features = torch.cat((embedding, noisy), dim = 0)
        # labels = torch.cat((data.y, data.y), dim = 0).to(device)
        
        pred = classifier(features)
        cls_loss = F.nll_loss(pred, labels)
        mi_loss = kl_loss
        
        optimizer.zero_grad()
        loss = cls_loss + args.con_weight * pos_penalty
        loss = loss + args.mi_weight * mi_loss
        loss.backward()
        
        total_loss += loss.item() * num_graphs
        optimizer.step()
    
    return total_loss / len(loader.dataset)


@torch.no_grad()
def vgib_eval(model, classifier, device, loader):
    model.eval()
    
    loss = 0
    y = []
    sub_preds, trivial_preds = [], []
    sub_pred_prob, trivial_pred_prob = [], []
    for data in loader:
        data = data.to(device)
        
        out, _, _, _, _, _ = model(data)
        out = classifier(out)
        pred = out.max(1)[1]
        
        y.append(data.y)
        sub_preds.append(pred)
        # trivial_preds.append(trivial_pred)
        sub_pred_prob.append(out[:, 1])
        # trivial_pred_prob.append(trivial_out[:, 1])
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    
    y = torch.cat(y).cpu().numpy()
    sub_preds = torch.cat(sub_preds).cpu().numpy()
    # trivial_preds = torch.cat(trivial_preds).cpu().numpy()
    sub_pred_prob = torch.cat(sub_pred_prob).cpu().numpy()
    # trivial_pred_prob = torch.cat(trivial_pred_prob).cpu().numpy()
    
    # sub_precision = precision_score(y, sub_preds, average = 'macro')
    subgraph_metric = {
        'accuracy': accuracy_score(y, sub_preds), 
        'precision': precision_score(y, sub_preds), 
        'recall': recall_score(y, sub_preds), 
        'f1': f1_score(y, sub_preds),
        'auc': roc_auc_score(y, sub_pred_prob)
    }
    
    # trivial_metric = {
    #     'accuracy': accuracy_score(y, trivial_preds), 
    #     'precision': precision_score(y, trivial_preds), 
    #     'recall': recall_score(y, trivial_preds), 
    #     'f1': f1_score(y, trivial_preds),
    #     'auc': roc_auc_score(y, trivial_pred_prob)
    # }
    
    return loss / len(loader.dataset), subgraph_metric, dict()
