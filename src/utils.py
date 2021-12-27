import numpy as np
import torch
from torch.utils.data import Dataset
import config
import copy
import dgl

def handle_data(inputData, train_len=None):
    len_data = [len(nowData) for nowData in inputData]
    if train_len is None:
        max_len = max(len_data)
    else:
        max_len = train_len
    # reverse the sequence
    us_pois = [list(reversed(upois)) + [0] * (max_len - le) if le < max_len else list(reversed(upois[-max_len:]))
               for upois, le in zip(inputData, len_data)]
    us_msks = [[1] * le + [0] * (max_len - le) if le < max_len else [1] * max_len
               for le in len_data]
    return us_pois, us_msks, max_len
    
def handle_adj(adj, sample_num):
    n_entity = len(adj)
    adj_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    for entity in range(1, n_entity):
        neighbor = list(adj[entity])
        n_neighbor = len(neighbor)
        if n_neighbor == 0:
            continue
        if n_neighbor == sample_num:
            sampled_indices = list(range(n_neighbor))
        elif n_neighbor > sample_num:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=False)
        elif sample_num % n_neighbor == 0:
            sampled_indices = []
            for i in range(sample_num//n_neighbor):
                sampled_indices += list(range(n_neighbor))
        else:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=True)
        adj_entity[entity] = np.array([neighbor[i] for i in sampled_indices])

    return adj_entity

class Data(Dataset):
    def __init__(self, data, edge2idx):
        self.edge2idx = edge2idx
        inputs, mask, max_len = handle_data(data[0], None)
        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(data[1])
        self.mask = np.asarray(mask)
        self.length = len(data[0])
        self.max_len = max_len

    def __getitem__(self, index):
        u_input, mask, target = self.inputs[index], self.mask[index], self.targets[index]
        
        # max_n_node = self.max_len
        # nodes = np.unique(u_input)
        # node2id = {n:i for i,n in enumerate(nodes)}
            
        #     adj_n[u][v] = adj_n[v][u] = 1
        # nodes = nodes.tolist() + (max_n_node - len(nodes)) * [0]
        # adj_n = np.eye(max_n_node)
        # adj_e = np.eye(5*(max_n_node-1))
        # adj_ne = np.zeros((max_n_node, 5*(max_n_node-1)))

        # edge2id = {}
        # cnt = 0
        # for i in range(len(u_input) - 1):
        #     u = node2id[u_input[i]]
        #     v = node2id[u_input[i+1]]
        #     adj_n[u][v] = adj_n[v][u] = 1

        # for i in range(len(u_input)-1,0,-1):
        #     for j in range(i-1, -1, -1):
        #         edge = (u_input[i], u_input[j])
        #         if  edge in self.edge2idx:
        #             if edge in edge2id:
        #                 w = edge2id[edge]
        #             else:
        #                 cnt += 1
        #                 w = edge2id[edge] = cnt
                        
        #             adj_ne[u][w] = 1
        #             break
        
        # edges = list(edge2id.keys())
        # for i in range(len(edges)-1):
        #     if edges[i] != 0:
        #         u = edge2id[edges[i]]
        #         for j in range(len(edges)):
        #             if edges[j] != 0:
        #                 v = edge2id[edges[j]]
        #                 if edges[1] == edges[0]:
        #                     adj_e[u][v] = adj_e[v][u] = 1

        # edges = (5*(max_n_node-1)) * [0]
        # for edge,id in edge2id.items():
        #     edges[id] = self.edge2idx[edge]

        # alias_inputs = [node2id[item] for item in u_input]

        return mask, u_input, target

    def __len__(self):
        return self.length