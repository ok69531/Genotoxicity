import torch
import numpy as np
# from .pgib_configures import model_args


def join_prototypes_by_activations(model, p, test_loader, device, cont, model_args):
 
    for idx, search_batch_input in enumerate(test_loader):
        search_batch = search_batch_input 

        with torch.no_grad():
            search_batch = search_batch.to(device)

            _, _, _, _, _, _, _, min_distance = model.forward(search_batch, merge=True) 
        if idx == 0:
            min_distances = min_distance 
        else:
            min_distances = torch.cat((min_distances, min_distance)) 
    
    distances_act = calculate_distances(torch.transpose(min_distances, 1, 0))

    assert 0 < p < 1
    ind = np.diag_indices(distances_act.shape[0]) 
    distances_act[ind[0], ind[1]] = np.inf 
    k = torch.kthvalue(distances_act.min(0)[0].cpu(), int(p * distances_act.cpu().size(0)))[0].item()
    dist_iterator = 0 
    no_of_prototypes = len(distances_act) 
    proto_joined = [] 
 
    distances = distances_act
    protos_ = np.arange(0, len(distances_act) ) 

    prot_num = list(range(no_of_prototypes))
    index_dic = {}
    for i in prot_num:
        index_dic[i] = i

    while dist_iterator < no_of_prototypes:  
        proto_distanses = distances[dist_iterator].cpu().detach().numpy() 
        if (proto_distanses <= k).any(): 
            to_join = np.argwhere(proto_distanses <= k)[:, 0] 
            
            model.model.last_layer.weight.data[:, dist_iterator] = \
                model.model.last_layer.weight.data[:, [dist_iterator, *to_join]].sum(1) 

            if cont == False:
                # prototype_predictor 
                dim = model.model.prototype_predictor.weight.data.shape[1]
                q_theta = model.model.prototype_predictor.weight.data.reshape(-1, dim , dim)

                for i in range(len(to_join)): 
                    reindex_to_join = index_dic[to_join[i]]
                    non_merge_list = np.delete(list(range(len(q_theta))), reindex_to_join) 

                    result = q_theta[non_merge_list].unsqueeze(0)            

                    index_dic.pop(to_join[i])
                    to_decrease = np.array(list(index_dic.keys())) > to_join[i]
                    iter = np.array(list(index_dic.keys()))[to_decrease]
                    for i in iter:
                        index_dic[i] = index_dic[i]-1

                    result = result.reshape(dim, -1)

                    model.model.prototype_predictor.weight = torch.nn.Parameter(torch.t(result))
            
            model.model.prototype_class_identity[dist_iterator] = \
                model.model.prototype_class_identity[[dist_iterator, *to_join], :].max(0)[0] 


            left_proto = np.setdiff1d(np.arange(model.model.last_layer.weight.data.shape[1]-model_args.hidden_dim), to_join) 
            joined = protos_[to_join] 
            protos_ = protos_[left_proto] 
            proto_joined.append([dist_iterator, joined]) 
            with torch.no_grad():
                last_layer = np.arange(model.model.last_layer.weight.data.shape[1])
                left_last_layer = np.concatenate((left_proto, last_layer[-model_args.hidden_dim:]))
                model.model.last_layer.weight = torch.nn.Parameter(model.model.last_layer.weight[:, left_last_layer])
                model.model.prototype_class_identity = model.model.prototype_class_identity[left_proto]
                model.model.prototype_vectors = torch.nn.Parameter(model.model.prototype_vectors[left_proto])
                distances = distances[np.ix_(left_proto, left_proto)] 

            no_of_prototypes = len(left_proto) 
        dist_iterator += 1

    model.model.num_prototypes = no_of_prototypes
    model.model.prototype_shape = model.model.prototype_vectors.shape
    print(f"prototypes after join: {no_of_prototypes}")
    return proto_joined


def calculate_distances(x):
    n, _ = x.shape
    x2 = torch.einsum('ij,ij->i', x, x) 
    y2 = x2.view(1, -1) 
    x2 = x2.view(-1, 1) 
    xy = torch.einsum('ij,kj->ik', x, x) 
    x2 = x2.repeat(1, n) 
    y2 = y2.repeat(n, 1) 
    norm2 = x2 - 2 * xy + y2 
    norm2 = norm2.abs()

    norm2[range(n), range(n)] = np.inf
    return norm2