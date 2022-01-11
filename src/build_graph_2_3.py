from math import e
import pickle
import argparse
from networkx import exception

from networkx.algorithms.shortest_paths import weighted
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
edge2fre = defaultdict(int)

for s in all_train_seq:
    for node in s:
        nodes.add(node)

for s in all_train_seq:
    for i in range(len(s)):
        edge = (s[i], s[i])
        edges.add(edge)

for s in all_train_seq:
    for i in range(len(s)-1):
        edge = (s[i], s[i+1])
        edges.add(edge)

for s in all_train_seq:
    for i in range(len(s)-1):
        for j in range(i+1, len(s)):
            edge = (s[i], s[j])
            edge2fre[edge] += 1
            
print(sorted(edge2fre.items(), key=lambda x:x[1], reverse=True)[100000:100010])
# exit()
print("nodes num:", len(nodes), "edges num:", len(edges), len(nodes)+len(edges))
pickle.dump(edge2idx, open('/home/xl/lxl/dataset/' + dataset + "/" + dataset + '/edge2idx.pkl',"wb"))
pickle.dump(edge2fre, open('/home/xl/lxl/dataset/' + dataset + "/" + dataset + '/edge2fre.pkl',"wb"))

G_n = nx.Graph()
for s in tqdm(all_train_seq):
    for i in range(0,len(s)-1):
        for j in range(i+1, len(s)):
            try:
                G_n[s[i]][s[i+1]]['weight']+=1
            except Exception:
                G_n.add_edge(s[i],s[i+1], weight=1)

print(len(G_n.nodes))

pickle.dump(G_n, open('/home/xl/lxl/model/DGL/data/'+dataset+'_adj.pkl',"wb"))
