import pdb
import numpy as np
from itertools import accumulate

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn.conv import GINConv
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


def get_readout_layers(readout):
    readout_func_dict = {
        "mean": global_mean_pool,
        "sum": global_add_pool,
        "max": global_max_pool
    }
    readout_func_dict = {k.lower(): v for k, v in readout_func_dict.items()}
    ret_readout = []
    for k, v in readout_func_dict.items():
        if k in readout.lower():
            ret_readout.append(v)
    return ret_readout


class GINNet(nn.Module):
    def __init__(self, output_dim, args, cont):
        super(GINNet, self).__init__()
        
        self.args = args
        self.cont = cont
        self.output_dim = output_dim
        self.latent_dim = args.hidden_dim
        self.num_gnn_layers = args.num_layers
        self.dense_dim = self.hidden_dim
        self.num_prototypes_per_class = args.num_prototypes_per_class
        
        self.readout_layers = get_readout_layers(args.readout)
        
        self.atom_encoder = AtomEncoder(emb_dim = self.latent_dim[0])
        
        self.gnn_layers = nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            self.gnn_layers.append(GINConv(self.latent_dim))

        self.fully_connected_1 = nn.Linear(self.latent_dim, self.latent_dim)
        self.fully_connected_2 = nn.Linear(self.latent_dim, 2)

        self.softmax = nn.Softmax(dim = -1)

        # prototype layers
        self.epsilon = 1e-4
        self.prototype_shape = (output_dim * self.num_prototypes_per_class, self.latent_dim)
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape, requires_grad = True))
        self.num_prototypes = self.prototype_shape[0]
        self.prototype_predictor = nn.Linear(self.latent_dim, self.num_prototypes * self.latent_dim, bias = False)
        self.mse_loss = nn.MSELoss()

        self.last_layer = nn.Linear(self.latent_dim + self.num_prototypes, output_dim, bias = False)

        assert (self.num_prototypes % output_dim == 0)

        self.prototype_class_identity = torch.zeros(self.num_prototypes, output_dim)
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1

        self.set_last_layer_incorrect_connection(incorrect_strength = -0.5)
    
    
    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight[:,: self.num_prototypes].data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)


    def gumbel_softmax(self, prob):
        return F.gumbel_softmax(prob, tau = 1, dim = -1)


    def prototype_distance(self, graph_emb):
        xp = torch.mm(graph_emb, torch.t(self.prototype_vectors))
        distance = -2 * xp + torch.sum(graph_emb ** 2, dim = 1, keepdim = True) + \
            torch.t(torch.sum(self.prototype_vectors ** 2, dim = 1, keepdim = True))
        similarity = torch.log((distance + 1) / (distance + self.epsilon))
        
        return similarity, distance
    
    
    def forward(self, data, merge = False):
        x = data.x if data.x is not None else data.feat
        edge_index, edge_attr, batch = data.edge_index, data.edge_attr, data.batch

        h = self.atom_encoder(x)
        for conv in self.gnn_layers:
            h = conv(h, edge_index, edge_attr)
        num_nodes = h.size(0)

        # this part is used to add noise
        node_feature = h
        node_emb = node_feature

        # this part is used to generate assignment matrix
        abstract_features_1 = torch.tanh(self.fully_connected_1(node_feature))
        assignment = F.softmax(self.fully_connected_2(abstract_features_1), dim = 1)

        gumbel_assignment = self.gumbel_softmax(assignment)

        # noisty graph representation
        lambda_pos = gumbel_assignment[:, 0].unsqueeze(dim = 1)
        lambda_neg = gumbel_assignment[:, 1].unsqueeze(dim = 1)

        # this is the graph embedding
        active = lambda_pos > 0.5
        active = active.squeeze()

        active_node_index = []
        node_number = [0]
        for i in range(batch[-1] + 1):
            node_number.append(len(batch[batch == i]))
        node_number = list(accumulate(node_number))

        for j in range(len(node_number) - 1):
            active_node_index.append(active[node_number[j] : node_number[j+1]].nonzero().squeeze().tolist())

        # KL loss
        static_node_feature = node_feature.clone().detach()
        node_feature_std, node_feature_mean = torch.std_mean(static_node_feature, dim = 0)
        node_feature_mean = node_feature_mean.repeat(num_nodes, 1)

        noisy_node_feature_mean = lambda_pos * node_feature + lambda_neg * node_feature_mean
        noisy_node_feature_std = lambda_neg * node_feature_std

        noisy_node_feature = noisy_node_feature_mean + \
            torch.rand_like(noisy_node_feature_mean) * noisy_node_feature_std

        for readout in self.readout_layers:
            noisy_graph_feature = readout(noisy_node_feature, batch)
        graph_emb = noisy_graph_feature

        epsilon = 0.0000001
        KL_tensor = 0.5 * ((noisy_node_feature_std ** 2) / (node_feature_std+epsilon) ** 2) + \
            torch.sum(((noisy_node_feature_mean -node_feature_mean)/(node_feature_std + epsilon))**2, dim = 0)
        KL_Loss = torch.mean(KL_tensor)

        Adj = to_dense_adj(edge_index, max_num_nodes = assignment.shape[0])[0]
        Adj.requires_grad = False

        try:
            torch.mm(torch.t(assignment),Adj)
        except:
            pdb.set_trace()
        new_adj = torch.mm(torch.t(assignment), Adj)
        new_adj = torch.mm(new_adj, assignment)
        normalize_new_adj = F.normalize(new_adj, p = 1, dim = 1)
        norm_diag = torch.diag(normalize_new_adj)

        # if torch.cuda.is_available():
        #     EYE = torch.ones(2).cuda()
        # else:
        #     EYE = torch.ones(2)
        EYE = torch.ones(2).to(edge_index.device)

        pos_penalty = self.mse_loss(norm_diag, EYE)

        ## graph embedding
        prototype_activations, min_distance = self.prototype_distance(graph_emb)

        final_embedding = torch.cat((prototype_activations, graph_emb), dim = 1)
        logits = self.last_layer(final_embedding)
        probs = self.softmax(logits)

        if self.cont:
            return logits, probs, active_node_index, graph_emb, KL_Loss, pos_penalty, prototype_activations, min_distance
        else:
            for i in range(graph_emb.shape[0]):
                predicted_prototype = self.prototype_predictor(torch.t(graph_emb[i])).reshape(-1, self.prototype_vectors.shape[1]) 
                if i == 0:
                    prototype_pred_losses = self.mse_loss(self.prototype_vectors, predicted_prototype).reshape(1)
                else:
                    prototype_pred_losses = torch.cat((prototype_pred_losses, self.mse_loss(self.prototype_vectors, predicted_prototype).reshape(1)))
            prototype_pred_loss = prototype_pred_losses.mean()

            return logits, probs, active_node_index, graph_emb, KL_Loss, pos_penalty, prototype_pred_loss, min_distance 



def get_model(output_dim, args, cont):
    # if model_args.model_name.lower() == 'gcn':
    #     return GCNNet(input_dim, output_dim, model_args)
    # elif model_args.model_name.lower() == 'gat':
    #     return GATNet(input_dim, output_dim, model_args)
    # elif model_args.model_name.lower() == 'gin':
    #     return GINNet(input_dim, output_dim, model_args)
    # else:
    #     raise NotImplementedError
    return GINNet(output_dim, args, cont)


class GnnBase(nn.Module):
    def __init__(self):
        super(GnnBase, self).__init__()

    def forward(self, data):
        # data = data.to(self.device)
        logits, prob, emb1, emb2, min_distances = self.model(data)
        return logits, prob, emb1, emb2, min_distances

    def update_state_dict(self, state_dict):
        original_state_dict = self.state_dict()
        loaded_state_dict = dict()
        for k, v in state_dict.items():
            if k in original_state_dict.keys():
                loaded_state_dict[k] = v
        self.load_state_dict(loaded_state_dict)

    def to_device(self):
        self.to(self.device)

    def save_state_dict(self):
        pass


class GnnNets(GnnBase):
    def __init__(self, output_dim, args, cont):
        super(GnnNets, self).__init__()
        self.model = get_model(output_dim, args, cont)
        # self.device = model_args.device

    def forward(self, data, merge=True):
        # data = data.to(self.device)
        logits, probs, active_node_index, graph_emb, KL_Loss, pos_penalty, sim_matrix, min_distance  = self.model(data, merge=True)
        return logits, probs, active_node_index, graph_emb, KL_Loss, pos_penalty, sim_matrix, min_distance


def warm_only(model):
    for p in model.model.gnn_layers.parameters():
        p.requires_grad = True
    model.model.prototype_vectors.requires_grad = True
    for p in model.model.last_layer.parameters():
        p.requires_grad = False


def joint(model):
    for p in model.model.gnn_layers.parameters():
        p.requires_grad = True
    model.model.prototype_vectors.requires_grad = True
    for p in model.model.last_layer.parameters():
        p.requires_grad = True


def pgib_train(model, optimizer, device, loader, criterion, epoch, args, cont):
    model.train()
    
    if epoch < args.warm_epochs:
        warm_only(model)
    else:
        joint(model)

    acc = []
    loss_list = []
    ld_loss_list = []

    for i, batch in enumerate(loader):
        batch = batch.to(device)
        
        if cont:
            logits, probs, active_node_index, graph_emb, KL_Loss, connectivity_loss, sim_matrix, _ = model(batch)
        else:
            logits, probs, active_node_index, graph_emb, KL_Loss, connectivity_loss, prototype_pred_loss, _ = model(batch)

        cls_loss = criterion(logits, batch.y)
        
        if cont:
            prototypes_of_correct_class = torch.t(model.model.prototype_class_identity.to(device)[:, batch.y])
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class
            positive_sim_matrix = sim_matrix * prototypes_of_correct_class
            negative_sim_matrix = sim_matrix * prototypes_of_wrong_class
            
            contrastive_loss = positive_sim_matrix.sum(dim = 1) / negative_sim_matrix.sum(dim = 1)
            contrastive_loss = - torch.log(contrastive_loss).mean()
        
        prototype_numbers = []
        for i in range(model.model.prototype_class_identity.shape[1]):
            prototype_numbers.append(int(torch.count_nonzero(model.model.prototype_class_identity[:, i])))
        prototype_numbers = accumulate(prototype_numbers)
        
        n = 0
        ld = 0
        
        for k in prototype_numbers:
            p = model.model.prototype_vectors[n:k]
            n = k
            p = F.normalize(p, p = 2, dim = 1)
            matrix1 = torch.mm(p, torch.t(p)) - torch.eye(p.shape[0]).to(device) - 0.3
            matrix2 = torch.zeros(matrix1.shape).to(device)
            ld += torch.sum(torch.where(matrix1 > 0, matrix1, matrix2))
            
        if cont:
            loss = cls_loss + args.alpha2 * contrastive_loss + args.con_weight * connectivity_loss + args.alpha1 * KL_Loss
        else:
            loss = cls_loss + args.alpha2 * prototype_pred_loss + args.con_weight * connectivity_loss + args.alpha1 * KL_Loss

        # optimization
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(model.model.parameters(), clip_value = 2.0)
        optimizer.step()
        
        # reocrd
        _, prediction = torch.max(logits, -1)
        loss_list.append(loss.item())
        ld_loss_list.append(ld.item())
        acc.append(prediction.eq(batch.y).cpu().numpy())

    # report train msg
    # print(f'Train Epoch: {epoch} | Loss: {np.average(loss_list):.3f} | Ld: {np.average(ld_loss_list):.3f} | '
    #     f'Acc: {np.concatenate(acc, axis = 0).mean():.3f}')
    
    return np.average(loss_list), np.average(ld_loss_list), np.concatenate(acc, axis = 0).mean()


@torch.no_grad()
def pgib_evaluate_GC(loader, model, device, criterion):
    model.eval()
    
    loss_list = []
    y = []
    sub_preds, trivial_preds = [], []
    sub_pred_prob, trivial_pred_prob = [], []
    for batch in loader:
        batch = batch.to(device)
        logits, probs, _, _, _, _, _, _ = model(batch)
        loss = criterion(logits, batch.y)
        
        _, prediction = torch.max(logits, -1)
        loss_list.append(loss.item())
        
        y.append(batch.y)
        sub_preds.append(prediction)
        sub_pred_prob.append(probs[:, 1])
    
    y = torch.cat(y).cpu().numpy()
    loss = np.average(loss_list)
    sub_preds = torch.cat(sub_preds).cpu().numpy()
    sub_pred_prob = torch.cat(sub_pred_prob).cpu().numpy()
    
    subgraph_metric = {
        'accuracy': accuracy_score(y, sub_preds), 
        'precision': precision_score(y, sub_preds), 
        'recall': recall_score(y, sub_preds), 
        'f1': f1_score(y, sub_preds),
        'auc': roc_auc_score(y, sub_pred_prob)
    }
    
    return loss, subgraph_metric, {}


@torch.no_grad()
def pgib_test_GC(loader, model, device, criterion):
    model.eval()
    
    acc = []
    loss_list = []
    pred_probs = []
    predictions = []
    
    with torch.no_grad():
        for _, batch in enumerate(loader):
            batch = batch.to(device)
            logits, probs, active_node_index, _, _, _, _, _ = model(batch)
            loss = criterion(logits, batch.y)
            
            # record
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            acc.append(prediction.eq(batch.y).cpu().numpy())
            predictions.append(prediction)
            pred_probs.append(probs)
    
    test_state = {
        'loss': np.average(loss_list),
        'acc': np.average(np.concatenate(acc, axis=0).mean())
    }
    
    pred_probs = torch.cat(pred_probs, dim=0).cpu().detach().numpy()
    predictions = torch.cat(predictions, dim=0).cpu().detach().numpy()
    
    return test_state, pred_probs, predictions
