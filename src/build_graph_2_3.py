import pickle
import argparse
from tqdm import tqdm
from collections import defaultdict
import numpy as np 
import networkx as nx
import sys
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='diginetica/Tmall/Nowplaying')
opt = parser.parse_args()

dataset = opt.dataset

all_train_seq = pickle.load(open('/home/xl/lxl/dataset/' + dataset + "/" + dataset + '/all_train_seq.txt', 'rb'))

nodes = set()
edges = set()
edge2idx = {}
for s in all_train_seq:
    for node in s:
        nodes.add(node)

for s in all_train_seq:
    for i in range(len(s)):
        edge = tuple(sorted([s[i], s[i]]))
        if edge not in edges:
                edges.add(edge)
                edge2idx[edge] = len(nodes) + len(edges)
    for i in range(len(s)-1):
        for j in range(i+1, len(s)):
            edge = tuple(sorted([s[i], s[j]]))
            if edge not in edges:
                edges.add(edge)
                edge2idx[edge] = len(nodes) + len(edges)

print("nodes num:", len(nodes), "edges num:", len(edges), len(nodes)+len(edges))
pickle.dump(edge2idx, open('/home/xl/lxl/dataset/' + dataset + "/" + dataset + '/edge2idx.pkl',"wb"))

# G_n = nx.Graph()
# for s in tqdm(all_train_seq):
#     for i in range(0,len(s)-1):
#         G_n.add_edge(s[i],s[i+1])
#         G_n.add_edge(s[i], edge2idx[(s[i],s[i+1])])
#         G_n.add_edge(s[i+1], edge2idx[(s[i],s[i+1])])
#         if i!=len(s)-2:
#             G_n.add_edge(edge2idx[(s[i],s[i+1])], edge2idx[(s[i+1],s[i+2])])

# G_n.add_node(0)
# for node in G_n.nodes:
#     G_n.add_edge(node,node)
# print(len(G_n.nodes))

# d = {"node_1":[], "node_2":[]}
# for edge in G_n.edges:
#     d["node_1"].append(edge[0])
#     d["node_2"].append(edge[1])

# data = pd.DataFrame(d)
# data.to_csv('../data/'+dataset+'_edges.csv',sep=',', index=False)