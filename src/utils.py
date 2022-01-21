from collections import defaultdict
import numpy as np
import random

from numpy.core.defchararray import add
from numpy.testing.utils import tempdir
import torch
from torch.utils.data import Dataset
import config
import copy
import dgl
import networkx as nx

# class Data(Dataset):
#     def __init__(self, data, edge2idx, edge2fre, adj, is_train=True):
#         self.edge2idx = edge2idx
#         self.edge2fre = edge2fre
#         self.adj = adj
#         self.is_train = is_train
#         inputs = [list(reversed(upois)) for upois in data[0]]
#         self.inputs = np.asarray(inputs)
#         self.targets = np.asarray(data[1])
#         self.length = len(data[0])
#         if is_train:
#             self.K=9
#         else:
#             self.K=9

#     def __getitem__(self, index):
#         seq, target = self.inputs[index],self.targets[index]

#         items = list(np.unique(seq))
#         item2id = {n:i for i,n in enumerate(items)}

#         add_seq = []
#         for item in items:
#             # if len(self.adj[item]) < 50:
#                 # add_seq += list(self.adj[item])
#             add_seq += [i[0] for i in dict(self.adj[item]).items() if i[1]['weight'] > 5]
#         add_items = [i for i in np.unique(add_seq) if i not in items]
#         for i,n in enumerate(add_items):
#             item2id[n] = i+len(items)
        
#         graph_data = {
#             ('item', 'interacts', 'item'):([],[]),
#             ('item', 'agg', 'target'):([],[])
#         }
#         g = dgl.heterograph(graph_data)
        
#         g = dgl.add_nodes(g, len(items)+len(add_items), ntype='item')
#         g.nodes['item'].data['iid'] = torch.tensor(items+add_items)
#         g.nodes['item'].data['pid'] = torch.tensor(list(range(len(items)))+[0]*len(add_items))

#         seq_nid = [item2id[item] for item in seq if item!= 0]
#         g.add_edges(seq_nid, seq_nid, {'dis': torch.zeros(len(seq_nid), dtype=torch.long)}, etype='interacts')
#         # adj = nx.Graph()
#         # adj.add_nodes_from(list(range(len(items))))
#         for i in range(1, self.K):
#             src = seq_nid[:-i]
#             dst = seq_nid[i:]
#             g.add_edges(src, dst, {'dis':i*torch.ones(len(src), dtype=torch.long)}, etype='interacts')
#             g.add_edges(dst, src, {'dis':i*torch.ones(len(src), dtype=torch.long)}, etype='interacts')

#         # if self.is_train:
#         #     for item in items:
#         #         # if len(self.adj[item]) < 50:
#         #         #     src = [item2id[item] for item in list(self.adj[item])] 
#         #         #     src = [item2id[i[0]] for i in dict(self.adj[item]) if i[1]['weight'] < 5]
#         #         #     g.add_edges(src, [item2id[item]]*len(src), etype='interacts')
                
#         #         src = [item2id[i[0]] for i in dict(self.adj[item]).items() if i[1]['weight'] > 5]
#         #         g.add_edges(src, [item2id[item]]*len(src), etype='interacts')

#         # for i in range(len(seq_nid)-1):
#         #     for j in range(i+1, i+2):
#         #         edge = tuple(sorted([seq[i], seq[j]]))
#         #         if edge in self.edge2fre and self.edge2fre[edge]>=1:
#         #             g.add_edges(seq_nid[i], seq_nid[j], {'dis':(j-i)*torch.ones(1, dtype=torch.long)}, etype='interacts')
#         #             g.add_edges(seq_nid[j], seq_nid[i], {'dis':(j-i)*torch.ones(1, dtype=torch.long)}, etype='interacts')
#         #         else:
#         #             exit()
        
#         # adj = nx.to_numpy_matrix(adj, nodelist=list(range(len(items))))
        
#         #agg
#         g = dgl.add_nodes(g, 1, ntype='target')
#         g.nodes['target'].data['tid'] = torch.tensor([0])
#         g.add_edges(seq_nid, [0]*len(seq_nid), etype='agg')
#         g.edges['agg'].data['pid'] = torch.tensor(range(len(seq_nid)))

#         return g, target

#     def __len__(self):
#         return self.length

class Data(Dataset):
    def __init__(self, data, edge2idx, edge2fre, adj, is_train=True):
        self.edge2idx = edge2idx
        self.edge2fre = edge2fre
        self.adj = adj
        self.is_train = is_train
        inputs = [list(reversed(upois)) for upois in data[0]]
        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(data[1])
        self.length = len(data[0])
        if is_train:
            self.K=2
        else:
            self.K=2

    def __getitem__(self, index):
        seq, target = self.inputs[index],self.targets[index]

        items = list(np.unique(seq))
        item2id = {n:i for i,n in enumerate(items)}

        graph_data = {
            ('item', 'interacts', 'item'):([],[]),
            ('item', 'agg', 'target'):([],[])
        }
        g = dgl.heterograph(graph_data)
        
        g = dgl.add_nodes(g, len(items), ntype='item')
        g.nodes['item'].data['iid'] = torch.tensor(items)
        g.nodes['item'].data['pid'] = torch.tensor(list(range(len(items))))
        is_last = np.zeros(len(items))
        is_last[item2id[seq[0]]] = 1
        g.nodes['item'].data['last'] = torch.tensor(is_last)

        seq_nid = [item2id[item] for item in seq if item!= 0]
        g.add_edges(seq_nid, seq_nid, {'dis': torch.zeros(len(seq_nid), dtype=torch.long)}, etype='interacts')

        for i in range(1, 2):
            src = seq_nid[:-i]
            dst = seq_nid[i:]
            g.add_edges(src, dst, {'dis':(i+1)*torch.ones(len(src), dtype=torch.long)}, etype='interacts')
            g.add_edges(dst, src, {'dis':i*torch.ones(len(src), dtype=torch.long)}, etype='interacts')
               
        #agg
        g = dgl.add_nodes(g, 1, ntype='target')
        g.nodes['target'].data['tid'] = torch.tensor([0])
        g.add_edges(seq_nid, [0]*len(seq_nid), etype='agg')
        g.edges['agg'].data['pid'] = torch.tensor(list(range(len(seq_nid))))

        return g, target

    def __len__(self):
        return self.length


import itertools

def create_index(sessions):
    lens = np.fromiter(map(len, sessions), dtype=np.long)
    session_idx = np.repeat(np.arange(len(sessions)), lens - 1)
    label_idx = map(lambda l: range(1, l), lens)
    label_idx = itertools.chain.from_iterable(label_idx)
    label_idx = np.fromiter(label_idx, dtype=np.long)
    idx = np.column_stack((session_idx, label_idx))
    return idx


class AugmentedDataset:
    def __init__(self, sessions, G, sort_by_length=False, NCE=False, training=False):
        self.sessions = sessions
        self.G = G
        # self.graphs = graphs
        index = create_index(self.sessions)  # columns: sessionId, labelIndex

        if sort_by_length:
            # sort by labelIndex in descending order
            ind = np.argsort(index[:, 1])[::-1]
            index = index[ind]
        
        if NCE:
            self.target2idx = defaultdict(list)
            for i, (sid,lidx) in enumerate(index):
                target = self.sessions[sid][lidx]
                self.target2idx[target].append(i)
        
        self.index = index

    def __getitem__(self, idx):
        #print(idx)
        sid, lidx = self.index[idx]
        seq = list(reversed(self.sessions[sid][:lidx]))
        target = self.sessions[sid][lidx]

        
        items = list(np.unique(seq))
        item2id = {n:i for i,n in enumerate(items)}

        graph_data = {
            ('item', 'interacts', 'item'):([],[]),
            ('item', 'agg', 'target'):([],[])
        }
        g = dgl.heterograph(graph_data)
        
        g = dgl.add_nodes(g, len(items), ntype='item')
        g.nodes['item'].data['iid'] = torch.tensor(items)
        g.nodes['item'].data['pid'] = torch.tensor(list(range(len(items))))
        is_last = np.zeros(len(items))
        is_last[item2id[seq[0]]] = 1
        g.nodes['item'].data['last'] = torch.tensor(is_last)

        seq_nid = [item2id[item] for item in seq]
        g.add_edges(list(set(seq_nid)), list(set(seq_nid)), {'dis': torch.zeros(len(list(set(seq_nid))), dtype=torch.long)}, etype='interacts')

        # EDGES = set([(item2id[e[0]], item2id[e[1]]) for e in self.G.subgraph(items).edges(data='weight') if e[2]>1]) | set([(item2id[e[1]], item2id[e[0]]) for e in self.G.subgraph(items).edges(data='weight') if e[2]>1])
        for i in range(1, 2):
            src = seq_nid[:-i]
            dst = seq_nid[i:]

            edges = set(zip(src,dst))
            if len(edges):
                src, dst = zip(*edges)
                g.add_edges(src, dst, {'dis':(2*i)*torch.ones(len(src), dtype=torch.long)}, etype='interacts')
                g.add_edges(dst, src, {'dis':(2*i-1)*torch.ones(len(src), dtype=torch.long)}, etype='interacts')
            # EDGES = EDGES - set(zip(src,dst)) - set(zip(dst,src)) - set(zip(seq_nid,seq_nid))
        
        # # print(seq, edges, self.G.subgraph(items).edges())
        # if len(EDGES) > 0:
        #     src, dst = zip(*EDGES)
        #     g.add_edges(src, dst, {'dis':100*torch.ones(len(src), dtype=torch.long)}, etype='interacts')

        #agg
        g = dgl.add_nodes(g, config.order, ntype='target')
        g.nodes['target'].data['tid'] = torch.tensor(list(range(config.order)))
        for i in range(config.order):
            if i==0 or i>=len(seq_nid):
                seq_nid_tmp = seq_nid
            else:
                seq_nid_tmp = seq_nid[:-i]
            
            tmp = [0]*len(seq_nid_tmp)
            for t in range(len(seq_nid_tmp)):
                tmp[t] = t
            g.add_edges(seq_nid_tmp, [i]*len(seq_nid_tmp), {'pid':torch.tensor(tmp)}, etype='agg')

        return g, target

    def __len__(self):
        return len(self.index)