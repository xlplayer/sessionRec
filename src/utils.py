import numpy as np
import random
import torch
from torch.utils.data import Dataset
import config
import copy
import dgl
import networkx as nx

class Data(Dataset):
    def __init__(self, data, edge2idx, edge2fre, is_train=True):
        self.edge2idx = edge2idx
        self.edge2fre = edge2fre
        inputs = [list(reversed(upois)) for upois in data[0]]
        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(data[1])
        self.length = len(data[0])
        if is_train:
            self.K=2
        else:
            self.K=3

    def __getitem__(self, index):
        seq, target = self.inputs[index],self.targets[index]

        items = np.unique(seq)
        item2id = {n:i for i,n in enumerate(items)}
        
        graph_data = {
            ('item', 'in', 'group'):([],[]),
            ('group', 'has', 'item'):([],[]),
            ('item', 'interacts', 'item'):([],[]),
            ('item', 'agg', 'target'):([],[])
        }
        g = dgl.heterograph(graph_data)
        
        g = dgl.add_nodes(g, len(items), ntype='item')
        g.nodes['item'].data['iid'] = torch.tensor(items)
        g.nodes['item'].data['pid'] = torch.arange(len(items), dtype=torch.long)

        seq_nid = [item2id[item] for item in seq if item!= 0]
        g.add_edges(seq_nid, seq_nid, {'dis': torch.zeros(len(seq_nid), dtype=torch.long)}, etype='interacts')
        # adj = nx.Graph()
        # adj.add_nodes_from(list(range(len(items))))
        for i in range(1, min(len(seq_nid), self.K)):
            src = seq_nid[:-i]
            dst = seq_nid[i:]
            g.add_edges(src, dst, {'dis':i*torch.ones(len(src), dtype=torch.long)}, etype='interacts')
            g.add_edges(dst, src, {'dis':i*torch.ones(len(src), dtype=torch.long)}, etype='interacts')

        # for i in range(len(seq_nid)-1):
        #     for j in range(i+1, i+2):
        #         edge = tuple(sorted([seq[i], seq[j]]))
        #         if edge in self.edge2fre and self.edge2fre[edge]>=1:
        #             g.add_edges(seq_nid[i], seq_nid[j], {'dis':(j-i)*torch.ones(1, dtype=torch.long)}, etype='interacts')
        #             g.add_edges(seq_nid[j], seq_nid[i], {'dis':(j-i)*torch.ones(1, dtype=torch.long)}, etype='interacts')
        #         else:
        #             exit()
        
        # adj = nx.to_numpy_matrix(adj, nodelist=list(range(len(items))))
        
        #agg
        g = dgl.add_nodes(g, 1, ntype='target')
        g.nodes['target'].data['tid'] = torch.tensor([0])
        g.add_edges(seq_nid, [0]*len(seq_nid), etype='agg')
        g.edges['agg'].data['pid'] = torch.tensor(range(len(seq_nid)))
        
        # #group
        # g = dgl.add_nodes(g, 1, ntype='group')
        # g.nodes['group'].data['gid'] = torch.tensor([0])
        # g.add_edges(seq_nid, [0]*len(seq_nid), etype='in')
        # g.add_edges([0]*len(seq_nid), seq_nid, etype='has')

        return g, target

    def __len__(self):
        return self.length
    