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

class Data(Dataset):
    def __init__(self, data, edge2idx=None, edge2fre=None, adj=None, is_train=True, unique=True, add_self_loop=True):
        self.unique = unique
        self.add_self_loop = add_self_loop
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
            ('item', 'agg', 'target'):([],[])
        }
        for i in range(config.window_size):
            graph_data[('item', 'interacts'+str(i), 'item')] = ([],[])

        g = dgl.heterograph(graph_data)
        
        g = dgl.add_nodes(g, len(items), ntype='item')
        g.nodes['item'].data['iid'] = torch.tensor(items)
        g.nodes['item'].data['pid'] = torch.tensor(list(range(len(items))))
        is_last = np.zeros((len(items), 3))
        is_last[item2id[seq[0]]][0] = 1
        is_last[item2id[seq[min(1,len(seq)-1)]]][1] = 1
        is_last[item2id[seq[min(2,len(seq)-1)]]][2] = 1
        g.nodes['item'].data['last'] = torch.tensor(is_last)

        seq_nid = [item2id[item] for item in seq if item!= 0]
        if self.add_self_loop:
            g.add_edges(list(set(seq_nid)), list(set(seq_nid)), {'dis': torch.zeros(len(list(set(seq_nid))), dtype=torch.long)}, etype='interacts')

        for i in range(config.window_size):
            src, dst = [], []
            for j in range(1, i+2):
                src = src + seq_nid[:-j]
                dst = dst + seq_nid[j:]
            if self.unique:
                edges = set(zip(src,dst))
            else:
                edges = list(zip(src,dst))
            if len(edges):
                src, dst = zip(*edges)
                g.add_edges(src, dst, {'dis':(i)*torch.ones(len(src), dtype=torch.long)}, etype='interacts'+str(i))
               
        #agg
        g = dgl.add_nodes(g, 1, ntype='target')
        g.nodes['target'].data['tid'] = torch.tensor([0])
        g.add_edges(seq_nid, [0]*len(seq_nid), etype='agg')
        g.edges['agg'].data['pid'] = torch.tensor(list(range(len(seq_nid))))
        g.edges['agg'].data['pid1'] = torch.tensor(list(range(len(seq_nid))))
        g.edges['agg'].data['pid2'] = torch.tensor(list(range(len(seq_nid))))

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
    def __init__(self, sessions, sort_by_length=False, NCE=False, training=False, epoch=None, train_len=None, unique=True, add_self_loop=True):
        self.add_self_loop = add_self_loop
        self.unique = unique
        self.training = training
        self.sessions = sessions
        self.train_len = train_len
        print(self.train_len)
        index = create_index(self.sessions)  # columns: sessionId, labelIndex
        self.index = index

    def __getitem__(self, idx):
        #print(idx)
        sid, lidx = self.index[idx]
        seq = list(reversed(self.sessions[sid][:lidx]))
        target = self.sessions[sid][lidx]

        
        items = list(np.unique(seq))
        item2id = {n:i for i,n in enumerate(items)}

        graph_data = {
            ('item', 'agg', 'target'):([],[])
        }
        for i in range(config.window_size):
            graph_data[('item', 'interacts'+str(i), 'item')] = ([],[])
        g = dgl.heterograph(graph_data)
        
        g = dgl.add_nodes(g, len(items), ntype='item')
        g.nodes['item'].data['iid'] = torch.tensor(items)
        g.nodes['item'].data['pid'] = torch.tensor(list(range(len(items))))
        is_last = np.zeros((len(items), 3))
        is_last[item2id[seq[0]]][0] = 1
        is_last[item2id[seq[min(1,len(seq)-1)]]][1] = 1
        is_last[item2id[seq[min(2,len(seq)-1)]]][2] = 1
        g.nodes['item'].data['last'] = torch.tensor(is_last)

        seq_nid = [item2id[item] for item in seq]
        if self.add_self_loop:
            g.add_edges(list(set(seq_nid)), list(set(seq_nid)), {'dis': torch.zeros(len(list(set(seq_nid))), dtype=torch.long)}, etype='interacts')
        
        for i in range(config.window_size):
            src, dst = [], []
            for j in range(1, i+2):
                src = src + seq_nid[:-j]
                dst = dst + seq_nid[j:]
            if self.unique:
                edges = set(zip(src,dst))
            else:
                edges = list(zip(src,dst))
            if len(edges):
                src, dst = zip(*edges)
                g.add_edges(src, dst, {'dis':(i)*torch.ones(len(src), dtype=torch.long)}, etype='interacts'+str(i))

        #agg
        g = dgl.add_nodes(g, 1, ntype='target')
        g.nodes['target'].data['tid'] = torch.tensor([0])
        g.add_edges(seq_nid, [0]*len(seq_nid), etype='agg')
        g.edges['agg'].data['pid'] = torch.tensor(list(range(len(seq_nid))))
        g.edges['agg'].data['pid1'] = torch.tensor(list(range(len(seq_nid))))
        g.edges['agg'].data['pid2'] = torch.tensor(list(range(len(seq_nid))))
        return g, target

    def __len__(self):
        return len(self.index)


class Mix_AugmentedDataset:
    def __init__(self, sessions, sort_by_length=False, NCE=False, training=False, epoch=None, train_len=None, unique=True, add_self_loop=True):
        self.add_self_loop = add_self_loop
        self.unique = unique
        self.training = training
        self.sessions = sessions
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
            # fre = np.mean([self.class_weight[self.cls2idx[item]] for item in self.sessions[sid][:lidx+1]])
            fre = self.class_weight[self.cls2idx[self.sessions[sid][lidx]]]
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
        class_weight = [(max_num / i)**2 for i in num_list]
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
            sample_index = random.choices(range(self.__len__()), cum_weights=self.cum_weights, k=1)[0]


        sid1, lidx1 = self.index[sample_index]
        seq_all1 = list(reversed(self.sessions[sid1][:lidx1]))
        target1 = self.sessions[sid1][lidx1]

        seqs = [seq_all0, seq_all1]
        gs = []

        for idx, seq in enumerate(seqs):
            items = list(np.unique(seq))
            item2id = {n:i for i,n in enumerate(items)}

            graph_data = {
                ('item', 'agg', 'target'):([],[])
            }
            for i in range(config.window_size):
                graph_data[('item', 'interacts'+str(i), 'item')] = ([],[])
            g = dgl.heterograph(graph_data)
            
            g = dgl.add_nodes(g, len(items), ntype='item')
            g.nodes['item'].data['iid'] = torch.tensor(items)
            g.nodes['item'].data['pid'] = torch.tensor(list(range(len(items))))
            is_last = np.zeros((len(items), 3))
            is_last[item2id[seq[0]]][0] = 1
            is_last[item2id[seq[min(1,len(seq)-1)]]][1] = 1
            is_last[item2id[seq[min(2,len(seq)-1)]]][2] = 1
            g.nodes['item'].data['last'] = torch.tensor(is_last)

            seq_nid = [item2id[item] for item in seq]
            if self.add_self_loop:
                g.add_edges(list(set(seq_nid)), list(set(seq_nid)), {'dis': torch.zeros(len(list(set(seq_nid))), dtype=torch.long)}, etype='interacts')
            
            for i in range(config.window_size):
                src, dst = [], []
                for j in range(1, i+2):
                    src = src + seq_nid[:-j]
                    dst = dst + seq_nid[j:]
                if self.unique:
                    edges = set(zip(src,dst))
                else:
                    edges = list(zip(src,dst))
                if len(edges):
                    src, dst = zip(*edges)
                    g.add_edges(src, dst, {'dis':(i)*torch.ones(len(src), dtype=torch.long)}, etype='interacts'+str(i))
                    
            #agg
            g = dgl.add_nodes(g, 1, ntype='target')
            g.nodes['target'].data['tid'] = torch.tensor([0])
            g.add_edges(seq_nid, [0]*len(seq_nid), etype='agg')
            g.edges['agg'].data['pid'] = torch.tensor(list(range(len(seq_nid))))
            g.edges['agg'].data['pid1'] = torch.tensor(list(range(len(seq_nid))))
            g.edges['agg'].data['pid2'] = torch.tensor(list(range(len(seq_nid))))

            gs.append(g)

        return gs[0], gs[1], target0, target1

    def __len__(self):
        return len(self.index)