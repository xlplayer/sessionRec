from networkx.generators.random_graphs import fast_gnp_random_graph
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import math
import pandas as pd
import networkx as nx
import dgl
from dgl import function as fn
from dgl.nn.pytorch import edge_softmax

class LocalAggregator(nn.Module):
    def __init__(self, dim, alpha):
        super(LocalAggregator, self).__init__()
        self.dim = dim

        self.p = nn.Parameter(torch.Tensor(1,self.dim))
        self.q = nn.Parameter(torch.Tensor(1,self.dim))
        self.r = nn.Parameter(torch.Tensor(1,self.dim))

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h_n, h_e, adj_n, adj_ne, adj_e):
        # y1 = torch.matmul(self.r*h_e, h_e.permute((0,2,1))) / math.sqrt(self.dim) 
        # y1 = self.leakyrelu(y1)
        # y1 = F.softmax(y1 * (1.0-adj_e)*(-1e9), dim=-1)
        # h_e = torch.matmul(y1*adj_e, h_e)

        x1 = torch.matmul(self.p*h_n, h_n.permute((0,2,1))) / math.sqrt(self.dim)
        x1 = self.leakyrelu(x1)
        x1 = F.softmax(x1 + (1.0-adj_n)*(-1e9), dim=-1)
        x1 = torch.matmul(x1*adj_n, h_n)

        x2 = torch.matmul(self.q*h_n, h_e.permute((0,2,1))) / math.sqrt(self.dim) 
        x2 = self.leakyrelu(x2)
        x2 = F.softmax(x2 + (1.0-adj_ne)*(-1e9), dim=-1)
        x2 = torch.matmul(x2*adj_ne, h_e)
        x = x1 + x2
        return x

        y1 = torch.matmul(self.r*h_e, h_e.permute((0,2,1))) / math.sqrt(self.dim) 
        y1 = self.leakyrelu(y1)
        y1 = F.softmax(y1 * (1.0-adj_e)*(-1e9), dim=-1)
        y1 = torch.matmul(y1*adj_e, h_e)

        y2 = torch.matmul(self.q*h_e, x.permute((0,2,1))) / math.sqrt(self.dim) 
        y2 = self.leakyrelu(y2)
        y2 = F.softmax(y2 + (1.0-adj_ne.permute((0,2,1))*(-1e9)), dim=-1)
        y2 = torch.matmul(y2*adj_ne.permute((0,2,1)), x)
        y = y1 + y2


        z1 = torch.matmul(self.p*x, x.permute((0,2,1))) / math.sqrt(self.dim) 
        z1 = self.leakyrelu(z1)
        z1 = F.softmax(z1 + (1.0-adj_n)*(-1e9), dim=-1)
        z1 = torch.matmul(z1*adj_n, x)

        z2 = torch.matmul(self.q*x, y.permute((0,2,1))) / math.sqrt(self.dim) 
        z2 = self.leakyrelu(z2)
        z2 = F.softmax(z2 + (1.0-adj_ne)*(-1e9), dim=-1)
        z2 = torch.matmul(z2*adj_ne, h_e)
        z = z1 + z2
        return z

def udf_u_muladd_v(edges):
    return {'e':edges.src['ft'] * edges.dst['ft'] + edges.src['ft'] + edges.dst['ft']}

def udf_u_cat_v(edges):
    return {'e':torch.cat([edges.src['ft'], edges.dst['ft']], dim=-1)}

def udf_agg(edges):
    return {'m':edges.src['ft']*torch.sum(edges.data['e'] * edges.dst['ft'], dim=-1, keepdim=True)}
    
class DglAggregator(nn.Module):
    def __init__(self, dim, alpha):
        super(DglAggregator, self).__init__()
        self.dim = dim

        self.p = nn.Linear(self.dim, 1, bias=False)
        self.q = nn.Linear(2*self.dim, self.dim, bias=False)
        self.glu1 = nn.Linear(config.dim, config.dim)

        self.leakyrelu = nn.LeakyReLU(alpha)

    # def forward(self, h_n, adj_n):
    #     with adj_n.local_scope():
    #         adj_n.ndata['ft'] = h_n
    #         # adj_n.apply_edges(fn.u_mul_v('ft','ft','e'))
    #         adj_n.apply_edges(udf_u_muladd_v)
    #         e = self.p(adj_n.edata['e'])
    #         e= self.leakyrelu(e)
    #         adj_n.edata['a'] = edge_softmax(adj_n, e)
    #         adj_n.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
    #         rst = adj_n.ndata['ft']
    #         return rst

    # def forward(self, h_n, adj_n):
    #     with adj_n.local_scope():
    #         adj_n.nodes['item'].data['ft'] = h_n
    #         adj_n.apply_edges(fn.u_mul_v('ft','ft','e'), etype='interacts')
    #         e = self.p(adj_n.edges['interacts'].data['e'])
    #         e= self.leakyrelu(e)
    #         adj_n.edges['interacts'].data['a'] = edge_softmax(adj_n['interacts'], e)
    #         adj_n.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'), etype='interacts')
    #         rst = adj_n.nodes['item'].data['ft']
    #         return rst

    def forward(self, h_n, h_p, h_t, adj_n):
        with adj_n.local_scope():

            ###item to item
            adj = adj_n.edge_type_subgraph(['interacts'])
            adj.ndata['ft'] = h_n
            adj.apply_edges(fn.u_mul_v('ft','ft','e'))
            e = self.p(adj.edata['e'])
            e= self.leakyrelu(e)
            adj.edata['a'] = edge_softmax(adj, e)
            adj.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            

            ###agg
            adj = adj_n.edge_type_subgraph(['agg'])
            adj.nodes['target'].data['ft'] = h_t
            adj.edges['agg'].data['pos'] = h_p
            adj.apply_edges(fn.copy_src('ft','ft'))
            e = self.q(torch.cat([adj.edata['ft'], adj.edata['pos']], dim=-1))
            adj.edata['e'] = torch.tanh(e)
            adj.update_all(udf_agg, fn.sum('m', 'ft'))

            return adj_n.nodes['target'].data['ft']


class SessionGraph(nn.Module):
    def __init__(self):
        super(SessionGraph, self).__init__()
        edges = pd.read_csv(config.graph_path)
        graph = nx.from_edgelist(edges.values.tolist())
        self.num_node = len(graph.nodes)
        print(self.num_node)

        self.embedding = nn.Embedding(self.num_node, config.dim)
        self.pos_embedding = nn.Embedding(200, config.dim)
        self.target_embedding = nn.Embedding(1, config.dim)
        
        # Parameters        
        self.local_agg = DglAggregator(config.dim, config.alpha)
        self.w_1 = nn.Parameter(torch.Tensor(2 * config.dim, config.dim))
        self.w_2 = nn.Parameter(torch.Tensor(config.dim, 1))
        self.glu1 = nn.Linear(config.dim, config.dim)
        self.glu2 = nn.Linear(config.dim, config.dim, bias=False)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)#, weight_decay=config.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.lr_dc_step, gamma=config.lr_dc)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(config.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        

    # def forward(self, alias_inputs, mask, adj_n):
    #     h_n = self.embedding(adj_n.nodes['item'].data['iid'])
    #     h_local = self.local_agg(h_n, adj_n)
    #     h_local = F.dropout(h_local, config.dropout_local, training=self.training)
    #     hidden = h_local
    #     get = lambda index: hidden[alias_inputs[index]]
    #     hidden = torch.stack([get(i) for i in torch.arange(alias_inputs.shape[0]).long()])
        
    #     mask = mask.unsqueeze(-1)
    #     batch_size = hidden.shape[0]
    #     len = hidden.shape[1]
    #     pos_emb = self.pos_embedding.weight[:len]
    #     pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

    #     hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
    #     hs = hs.unsqueeze(-2).repeat(1, len, 1)
    #     nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
    #     nh = torch.tanh(nh)
    #     nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
    #     beta = torch.matmul(nh, self.w_2)
    #     beta = beta * mask
    #     select = torch.sum(beta * hidden, 1)
    #     b = self.embedding.weight[1:config.num_node]  # n_nodes x latent_size
    #     scores = torch.matmul(select, b.transpose(1, 0))
    #     return scores
    
    def forward(self, alias_inputs, mask, adj_n):
        h_n = self.embedding(adj_n.nodes['item'].data['iid'])
        h_p = self.pos_embedding(adj_n.edges['agg'].data['pid'])
        h_r = self.target_embedding(adj_n.nodes['target'].data['tid'])
        select = self.local_agg(h_n, h_p, h_r, adj_n)
        b = self.embedding.weight[1:config.num_node]  # n_nodes x latent_size
        scores = torch.matmul(select, b.transpose(1, 0))
        return scores