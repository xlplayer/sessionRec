import numpy as np
import torch
from torch.utils.data import Dataset
import config
import copy
import dgl

class Data(Dataset):
    def __init__(self, data, edge2idx):
        self.edge2idx = edge2idx
        inputs = [list(reversed(upois)) for upois in data[0]]
        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(data[1])
        self.length = len(data[0])

    def __getitem__(self, index):
        seq, target = self.inputs[index],self.targets[index]

        items = np.unique(seq)
        item2id = {n:i for i,n in enumerate(items)}
        graph_data = {
            ('item', 'interacts', 'item'):([],[]),
            ('item', 'agg', 'target'):([],[])
        }
        g = dgl.heterograph(graph_data)
        g = dgl.add_nodes(g, len(items), ntype='item')
        g.nodes['item'].data['iid'] = torch.tensor(items)

        seq_nid = [item2id[item] for item in seq if item!= 0]
        g.add_edges(seq_nid, seq_nid, etype='interacts')
        src = seq_nid[:-1]
        dst = seq_nid[1:]
        g.add_edges(src, dst, etype='interacts')
        g.add_edges(dst, src, etype='interacts')
        
        g = dgl.add_nodes(g, 1, ntype='target')
        g.nodes['target'].data['tid'] = torch.tensor([0])
        g.add_edges(seq_nid, [0]*len(seq_nid), etype='agg')
        g.edges['agg'].data['pid'] = torch.tensor(range(len(seq_nid)))
        
        return g, target

    def __len__(self):
        return self.length
    