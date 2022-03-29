from collections import defaultdict
import numpy as np
import random
from itertools import accumulate

from numpy.core.defchararray import add
from numpy.testing.utils import tempdir
import torch
from torch.utils.data import Dataset
import config
import copy
import dgl
import networkx as nx
from tqdm import tqdm

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
    def __init__(self, sessions, G, sort_by_length=False, NCE=False, training=False, epoch=None, train_len=None):
        # if training and epoch is not None and epoch == 0:
        #     print("hhh")            
        #     sessions = np.array([list(reversed(s)) for s in sessions[-150000:]])
        self.training = training
        self.sessions = sessions
        self.G = G
        self.train_len = train_len
        print(self.train_len)
        # self.graphs = graphs
        index = create_index(self.sessions)  # columns: sessionId, labelIndex

        if sort_by_length:
            # sort by labelIndex in descending order
            ind = np.argsort(index[:, 1])[::-1]
            index = index[ind]
        
        # if training:
        #     self.target2idx = defaultdict(list)
        #     for i, (sid,lidx) in enumerate(index):
        #         for item in self.sessions[sid][:lidx+1]:
        #             if len(self.target2idx[item])==0 or self.target2idx[item][-1] != i:
        #                 self.target2idx[item].append(i)

        #     flags = [0] * len(index)
        #     for v in tqdm(self.target2idx.values()):
        #         l = max(100, len(v)//2)
        #         for id in set(v[-l:]):
        #             flags[id] = 1

        #     ind = []
        #     for i, (sid,lidx) in enumerate(index):
        #         if flags[i]:
        #             ind.append(i)
        #     index = index[ind]

        # if not training:
        #     y = [self.sessions[sid][lidx] for sid,lidx in index]
        #     ind = np.argsort(y)
        #     of = open("log_test.txt", 'w')
        #     for i in ind:
        #         sid, lidx = index[i]
        #         print(self.sessions[sid][:lidx], self.sessions[sid][lidx], file=of)
        #     exit()
        
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
            ('item', 'agg', 'target'):([],[]),
            ('target', 'overlap', 'target'):([],[])
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
        
        # if self.training:
        #     threshold = self.train_len
        # else:
        #     threshold = self.train_len

        # EDGES = set([(item2id[e[0]], item2id[e[1]]) for e in self.G.subgraph(items).edges(data='time')])# if e[2]<threshold]) 
        for i in [1]:
            src = seq_nid[:-i]
            dst = seq_nid[i:]

            edges = set(zip(src,dst))
            if len(edges):
                src, dst = zip(*edges)
                g.add_edges(src, dst, {'dis':(2*i)*torch.ones(len(src), dtype=torch.long)}, etype='interacts')
                g.add_edges(dst, src, {'dis':(2*i-1)*torch.ones(len(src), dtype=torch.long)}, etype='interacts')
                
            # EDGES = EDGES - set(zip(src,dst)) - set(zip(dst,src)) - set(zip(seq_nid,seq_nid))
        
        # # # print(seq, edges, self.G.subgraph(items).edges())
        # if not self.training:
        #     if len(EDGES) > 0:
        #         src, dst = zip(*EDGES)
        #         g.add_edges(src, dst, {'dis':100*torch.ones(len(src), dtype=torch.long)}, etype='interacts')

        #agg
        g = dgl.add_nodes(g, 1, ntype='target')
        g.nodes['target'].data['tid'] = torch.tensor([0])
        g.add_edges(seq_nid, [0]*len(seq_nid), {'pid':torch.tensor(list(range(len(seq_nid))))}, etype='agg')
        return g, target

    def __len__(self):
        return len(self.index)


class Pretrain_AugmentedDataset:
    def __init__(self, sessions):
        self.sessions = sessions
        index = create_index(self.sessions)  # columns: sessionId, labelIndex
        self.index = index

    def __getitem__(self, idx):
        #print(idx)
        sid, lidx = self.index[idx]
        seq_all = list(reversed(self.sessions[sid][:lidx+1]))

        seqs = [seq_all[:-1], seq_all[1:]]
        gs = []
        for i in range(2):
            seq = seqs[i]
            items = list(np.unique(seq))
            item2id = {n:i for i,n in enumerate(items)}

            graph_data = {
                ('item', 'interacts', 'item'):([],[]),
                ('item', 'agg', 'target'):([],[]),
                ('target', 'overlap', 'target'):([],[])
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
        
            for i in [1]:
                src = seq_nid[:-i]
                dst = seq_nid[i:]

                edges = set(zip(src,dst))
                if len(edges):
                    src, dst = zip(*edges)
                    g.add_edges(src, dst, {'dis':(2*i)*torch.ones(len(src), dtype=torch.long)}, etype='interacts')
                    g.add_edges(dst, src, {'dis':(2*i-1)*torch.ones(len(src), dtype=torch.long)}, etype='interacts')    

            #agg
            g = dgl.add_nodes(g, 1, ntype='target')
            g.nodes['target'].data['tid'] = torch.tensor([0])
            pos = np.array(range(len(seq_nid)))
            g.add_edges(seq_nid, [0]*(len(seq_nid)), {'pid':torch.tensor(pos)}, etype='agg')
            
            gs.append(g)

        return gs[0], gs[1]

    def __len__(self):
        return len(self.index)


class Two_AugmentedDataset:
    def __init__(self, sessions, G, sort_by_length=False, NCE=False, training=False, epoch=None, train_len=None):
        self.training = training
        self.sessions = sessions
        self.G = G
        self.train_len = train_len
        print(self.train_len)
        index = create_index(self.sessions)  # columns: sessionId, labelIndex

        if sort_by_length:
            # sort by labelIndex in descending order
            ind = np.argsort(index[:, 1])[::-1]
            index = index[ind]

        self.index = index

    def __getitem__(self, idx):
        #print(idx)
        sid, lidx = self.index[idx]
        seq_all = list(reversed(self.sessions[sid][:lidx]))
        target = self.sessions[sid][lidx]

        seqs = [seq_all, seq_all]
        gs = []

        for idx, seq in enumerate(seqs):
            items = list(np.unique(seq))
            item2id = {n:i for i,n in enumerate(items)}

            graph_data = {
                ('item', 'interacts', 'item'):([],[]),
                ('item', 'agg', 'target'):([],[]),
                ('target', 'overlap', 'target'):([],[])
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
            
            for i in [1]:
                src = seq_nid[:-i]
                dst = seq_nid[i:]
                edges = set(zip(src,dst))
                if len(edges):
                    src, dst = zip(*edges)
                    g.add_edges(src, dst, {'dis':(2*i)*torch.ones(len(src), dtype=torch.long)}, etype='interacts')
                    g.add_edges(dst, src, {'dis':(2*i-1)*torch.ones(len(src), dtype=torch.long)}, etype='interacts')  
                    
            #agg
            g = dgl.add_nodes(g, 1, ntype='target')
            g.nodes['target'].data['tid'] = torch.tensor([0])
            g.add_edges(seq_nid, [0]*len(seq_nid), {'pid':torch.tensor([i for i in range(len(seq_nid))] )}, etype='agg')

            gs.append(g)

        return gs[0], gs[1], target

    def __len__(self):
        return len(self.index)



class Mix_AugmentedDataset:
    def __init__(self, sessions, G, sort_by_length=False, NCE=False, training=False, epoch=None, train_len=None):
        self.training = training
        self.sessions = sessions
        self.G = G
        self.train_len = train_len
        print(self.train_len)
        index = create_index(self.sessions)  # columns: sessionId, labelIndex

        if sort_by_length:
            # sort by labelIndex in descending order
            ind = np.argsort(index[:, 1])[::-1]
            index = index[ind]

        self.index = index

        self.class_dict = defaultdict(list)
        for idx in range(self.__len__()):
            sid, lidx = self.index[idx]
            for item in self.sessions[sid][:lidx+1]:
                self.class_dict[item].append(idx)
 
        self.cls2idx = {c:i for i,c in enumerate(list(self.class_dict.keys()))}
        self.idx2cls = {i:c for i,c in enumerate(list(self.class_dict.keys()))}
        self.class_weight, self.sum_weight = self.get_weight(self.get_annotations())

        weights = []
        for idx in range(self.__len__()):
            sid, lidx = self.index[idx]
            fre = np.mean([self.class_weight[self.cls2idx[item]] for item in self.sessions[sid][:lidx+1]])
            weights.append(fre)
        self.cum_weights = list(accumulate(weights))
        

    def sample_class_index_by_weight(self):
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(len(self.idx2cls)):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return self.idx2cls[i]

    def get_annotations(self):
        annos = []
        for sess in self.sessions:
            for item in sess:
                annos.append({'category_id': int(item)})
        return annos

    def get_weight(self, annotations):
        num_list = [0] * len(self.cls2idx)
        for anno in annotations:
            category_id = anno["category_id"]
            num_list[self.cls2idx[category_id]] += 1
        max_num = max(num_list)
        class_weight = [max_num / i for i in num_list]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight

    def __getitem__(self, idx):
        #print(idx)
        sid0, lidx0 = self.index[idx]
        seq_all0 = list(reversed(self.sessions[sid0][:lidx0]))
        target0 = self.sessions[sid0][lidx0]

        TYPE = "mean"
        if TYPE == "reverse":
            sample_class = self.sample_class_index_by_weight()
            sample_indexes = self.class_dict[sample_class]
            sample_index = random.choice(sample_indexes)
        elif TYPE == "balance":
            sample_class = self.idx2cls[random.randint(0, len(self.idx2cls)-1)]
            sample_indexes = self.class_dict[sample_class]
            sample_index = random.choice(sample_indexes)
        elif TYPE == "uniform":
            sample_index = random.randint(0, self.__len__() - 1)
        elif TYPE == "mean":
            sample_index = random.choices(self, range(self.__len__), cum_weights=self.cum_weights, k=1)[0]


        sid1, lidx1 = self.index[sample_index]
        seq_all1 = list(reversed(self.sessions[sid1][:lidx1]))
        target1 = self.sessions[sid1][lidx1]

        seqs = [seq_all0, seq_all1]
        gs = []

        for idx, seq in enumerate(seqs):
            items = list(np.unique(seq))
            item2id = {n:i for i,n in enumerate(items)}

            graph_data = {
                ('item', 'interacts', 'item'):([],[]),
                ('item', 'agg', 'target'):([],[]),
                ('target', 'overlap', 'target'):([],[])
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
            
            for i in [1]:
                src = seq_nid[:-i]
                dst = seq_nid[i:]
                edges = set(zip(src,dst))
                if len(edges):
                    src, dst = zip(*edges)
                    g.add_edges(src, dst, {'dis':(2*i)*torch.ones(len(src), dtype=torch.long)}, etype='interacts')
                    g.add_edges(dst, src, {'dis':(2*i-1)*torch.ones(len(src), dtype=torch.long)}, etype='interacts')  
                    
            #agg
            g = dgl.add_nodes(g, 1, ntype='target')
            g.nodes['target'].data['tid'] = torch.tensor([0])
            g.add_edges(seq_nid, [0]*len(seq_nid), {'pid':torch.tensor([i for i in range(len(seq_nid))] )}, etype='agg')

            gs.append(g)

        return gs[0], gs[1], target0, target1

    def __len__(self):
        return len(self.index)