from turtle import forward
from networkx.generators.random_graphs import fast_gnp_random_graph
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
import config
import math
import pandas as pd
import networkx as nx
import numpy as np
import copy
import dgl
from dgl import function as fn
from dgl.nn.pytorch import edge_softmax
from entmax import sparsemax, entmax15, entmax_bisect
from label_smooth import LabelSmoothSoftmaxCEV1


def udf_agg(edges):
    return {'m':edges.src['ft']*torch.sum(edges.data['e'] * edges.dst['ft'], dim=-1, keepdim=True)}

def dis_u_mul_v(edges):
    return {'e': edges.src['ft']*edges.dst['ft']*edges.data['d'],'mask':torch.cat([edges.src['ft']*edges.dst['ft'], edges.data['d']], dim=-1)}
    return {'e':edges.src['ft']*edges.dst['ft']*edges.data['d'], 'sim': torch.sigmoid(torch.sum(edges.src['ft']*edges.dst['ft'], dim=-1))}
    # print(edges.data['edot'].shape, edges.data['dis'].shape, edges.dst['d'].shape)
    edges.data['dis'] = edges.data['dis'].unsqueeze(-1)
    return {'d_fea': torch.cat([edges.data['d'], edges.src['ft']*edges.dst['ft']], dim=-1), 'sim':torch.sum(edges.src['ft']*edges.dst['ft'], dim=-1)}

def udf_group(nodes):
    # print(nodes.mailbox['m'].shape, nodes.mailbox['m'].shape[1], nodes.data['ft'].shape, nodes.data['ft'].unsqueeze(-1).repeat(1,nodes.mailbox['m'].shape[1],1).shape)
    sim = torch.sum(nodes.mailbox['m'] * nodes.data['ft'].unsqueeze(-2).repeat(1,nodes.mailbox['m'].shape[1],1), -1)
    sim = F.softmax(sim, dim=-1)
    return {'ft':torch.sum(sim.unsqueeze(-1)*nodes.mailbox['m'], dim=-2)}

def udf_message(edges):
    return {'ft': edges.src['ft'], 'gumble':edges.data['gumble'], 'sim':edges.data['sim']}

def udf_edge_softmax(nodes):
    # print(nodes.mailbox['ft'].shape, nodes.mailbox['gumble'].shape, nodes.mailbox['sim'].shape)
    # exit()
    a = F.softmax(nodes.mailbox['sim'], dim=-1) *  nodes.mailbox['gumble'][:,:,1]
    a = a.unsqueeze(-1) 
    return {'ft': torch.sum(nodes.mailbox['ft']*a, dim=1)}

def src_dot_dst(src_field, dst_field, out_field):
    """
    This function serves as a surrogate for `src_dot_dst` built-in apply_edge function.
    """
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}
    return func

def scaled_exp(field, c):
    """
    This function applies $exp(x / c)$ for input $x$, which is required by *Scaled Dot-Product Attention* mentioned in the paper.
    """
    def func(edges):
        return {field: torch.exp((edges.data[field] / c).clamp(-10, 10))}
    return func

def clones(module, k):
    return nn.ModuleList(
        copy.deepcopy(module) for _ in range(k)
    )

class NoamOpt(object):
    def __init__(self, model_size, factor, warmup, optimizer):
        """
        model_size: hidden size
        factor: coefficient
        warmup: warm up steps(step ** (-0.5) == step * warmup ** (-1.5) holds when warmup equals step)
        """
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5))
            )

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()


class MultiHeadAttention(nn.Module):
    "Multi-Head Attention"
    def __init__(self, h, dim_model):
        "h: number of heads; dim_model: hidden dimension"
        super(MultiHeadAttention, self).__init__()
        self.d_k = dim_model // h
        self.h = h
        # W_q, W_k, W_v, W_o
        self.linears = clones(
            nn.Linear(dim_model, dim_model, bias=False), 4
        )

    def get(self, x, fields='qkv'):
        "Return a dict of queries / keys / values."
        batch_size = x.shape[0]
        ret = {}
        if 'q' in fields:
            ret['q'] = x.view(batch_size, self.h, self.d_k)
        if 'k' in fields:
            ret['k'] = x.view(batch_size, self.h, self.d_k)
        if 'v' in fields:
            ret['v'] = x.view(batch_size, self.h, self.d_k)
        return ret

    def get_o(self, x):
        "get output of the multi-head attention"
        batch_size = x.shape[0]
        return x.view(batch_size, -1)
        return self.linears[3](x.view(batch_size, -1))

class PositionwiseFeedForward(nn.Module):
    '''
    This module implements feed-forward network(after the Multi-Head Network) equation:
    FFN(x) = max(0, x @ W_1 + b_1) @ W_2 + b_2
    '''
    def __init__(self, dim_model, dim_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(dim_model, dim_ff)
        self.w_2 = nn.Linear(dim_ff, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))

class PositionalEncoding(nn.Module):
    "Position Encoding module"
    def __init__(self, dim_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, dim_model, dtype=torch.float)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2, dtype=torch.float) *
                             -(np.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # Not a parameter but should be in state_dict

    def forward(self, pos):
        return torch.index_select(self.pe, 1, pos).squeeze(0)

class SubLayerWrapper(nn.Module):
    '''
    The module wraps normalization, dropout, residual connection into one equation:
    sublayerwrapper(sublayer)(x) = x + dropout(sublayer(norm(x)))
    '''
    def __init__(self, size, dropout):
        super(SubLayerWrapper, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn # (key, query, value, mask)
        self.feed_forward = feed_forward
        self.sublayer = clones(SubLayerWrapper(size, dropout), 2)

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.N = N
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def pre_func(self, i, fields='qkv'):
        layer = self.layers[i]
        def func(nodes):
            x = nodes.data['x']
            norm_x = layer.sublayer[0].norm(x)
            return layer.self_attn.get(norm_x, fields=fields)
        return func

    def post_func(self, i):
        layer = self.layers[i]
        def func(nodes):
            x, wv, z = nodes.data['x'], nodes.data['wv'], nodes.data['z']
            o = layer.self_attn.get_o(wv / z)
            x = x + o
            # x = x + layer.sublayer[0].dropout(o)
            # x = layer.sublayer[1](x, layer.feed_forward)
            return {'x': x if i < self.N - 1 else self.norm(x)}
        return func

class Embeddings(nn.Module):
    "Word Embedding module"
    def __init__(self, vocab_size, dim_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, dim_model)
        self.dim_model = dim_model

    def forward(self, x):
        return self.lut(x) * np.sqrt(self.dim_model)

class Transformer(nn.Module):
    def __init__(self, num_node, N=1, h=1, dim_ff=config.dim, dim_model=config.dim, dropout=0.1):
        super(Transformer, self).__init__()
        c = copy.deepcopy
        attn = MultiHeadAttention(h, dim_model)
        ff = PositionwiseFeedForward(dim_model, dim_ff)
        pos_enc = PositionalEncoding(dim_model, dropout)
        self.encoder = Encoder(EncoderLayer(dim_model, c(attn), c(ff), dropout), N)
        self.src_embed = Embeddings(num_node, dim_model)
        self.pos_enc = pos_enc
        self.h, self.d_k = h, dim_model//h
        self.pos_embedding = nn.Embedding(200, dim_model)
        self.target_embedding = nn.Embedding(1, dim_model)
        self.q = nn.Linear(2*dim_model, dim_model, bias=False)
        self.r = nn.Linear(2*dim_model, dim_model, bias=False)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)#, weight_decay=config.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.lr_dc_step, gamma=config.lr_dc)
        # self.optimizer = NoamOpt(dim_model, 0.1, 4000, torch.optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9))

        stdv = 1.0 / math.sqrt(config.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('k', 'q', 'score'))
        g.apply_edges(scaled_exp('score', np.sqrt(self.d_k)))
        # Send weighted values to target nodes
        g.send_and_recv(g.edges(), fn.src_mul_edge('v', 'score', 'v'), fn.sum('v', 'wv'))
        g.send_and_recv(g.edges(), fn.copy_edge('score', 'score'), fn.sum('score', 'z'))

    def update_graph(self, g, pre_pairs, post_pairs):
        "Update the node states and edge states of the graph."

        # Pre-compute queries and key-value pairs.
        for pre_func in pre_pairs:
            g.apply_nodes(pre_func)
        self.propagate_attention(g)
        # Further calculation after attention mechanism
        for post_func in post_pairs:
            g.apply_nodes(post_func)

    def forward(self, G):
        with G.local_scope():
            g = G.edge_type_subgraph(['interacts'])
            # embed
            src_embed, src_pos = self.src_embed(g.ndata['iid']), self.pos_enc(g.ndata['pid'])
            g.ndata['x'] = self.pos_enc.dropout(src_embed) # + src_pos)

            for i in range(self.encoder.N):
                pre_func = self.encoder.pre_func(i, 'qkv')
                post_func = self.encoder.post_func(i)
                self.update_graph(g,[pre_func], [post_func])


            ###agg
            g = G.edge_type_subgraph(['agg'])
            g.nodes['item'].data['ft'] = g.nodes['item'].data['x']
            g.nodes['target'].data['ft'] = self.target_embedding(g.nodes['target'].data['tid'])
            g.edges['agg'].data['pos'] = self.pos_embedding(g.edges['agg'].data['pid'])
            g.apply_edges(fn.copy_src('ft','ft'))
            e = self.q(torch.cat([g.edata['ft'], g.edata['pos']], dim=-1))
            g.edata['e'] = torch.tanh(e)

            g.update_all(fn.copy_edge('ft', 'm'), fn.mean('m', 'mean'))
            f = self.r(torch.cat([g.nodes['target'].data['ft'], g.nodes['target'].data['mean']], dim=-1))
            g.nodes['target'].data['ft'] = f

            g.update_all(udf_agg, fn.sum('m', 'ft'))

            ###predict
            select = g.nodes['target'].data['ft']
            b = self.src_embed.lut.weight[1:config.num_node]  # n_nodes x latent_size
            scores = torch.matmul(select, b.transpose(1, 0))
            return scores

KMEAN_INIT_ITERS = 10

from inspect import isfunction
def default(x, d):
    if not exists(x):
        return d if not isfunction(d) else d()
    return x

def exists(val):
    return val is not None

# kmeans related function and class
def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]

def update_kmeans_on_backwards(module):
    module.kmean_modules = find_modules(module, Kmeans)
    def hook(_, grad_in, grad_out):
        for m in module.kmean_modules:
            m.update()

    return module.register_backward_hook(hook)

def similarity(x, means):
    return torch.einsum('bhld,hcd->bhlc', x, means)

def dists_and_buckets(x, means):
    dists = similarity(x, means)
    _, buckets = torch.max(dists, dim=-1)
    return dists, buckets

def batched_bincount(index, num_classes, dim=-1):
    shape = list(index.shape)
    shape[dim] = num_classes
    out = index.new_zeros(shape)
    out.scatter_add_(dim, index, torch.ones_like(index, dtype=index.dtype))
    return out

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def kmeans_iter(x, means, buckets = None):
    b, h, l, d, dtype, num_clusters = *x.shape, x.dtype, means.shape[1]

    if not exists(buckets):
        _, buckets = dists_and_buckets(x, means)

    bins = batched_bincount(buckets, num_clusters).sum(0, keepdim=True)
    zero_mask = bins.long() == 0

    means_ = buckets.new_zeros(b, h, num_clusters, d, dtype=dtype)
    means_.scatter_add_(-2, expand_dim(buckets, -1, d), x)
    means_ = F.normalize(means_.sum(0, keepdim=True), dim=-1).type(dtype)

    means = torch.where(zero_mask.unsqueeze(-1), means, means_)
    means = means.squeeze(0)
    return means

def distribution(dists, window_size):
    _, topk_indices = dists.topk(k=window_size, dim=-2)
    indices = topk_indices.transpose(-2, -1)
    return indices.reshape(*indices.size()[:2], -1)

def is_empty(t):
    return t.nelement() == 0

def ema(old, new, decay):
    if not exists(old):
        return new
    return old * decay + new * (1 - decay)
    
def ema_inplace(moving_avg, new, decay):
    if is_empty(moving_avg):
        moving_avg.data.copy_(new)
        return
    moving_avg.data.mul_(decay).add_(new, alpha= (1 - decay))

def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(2, expand_dim(indices, -1, last_dim))

class Kmeans(nn.Module):
    def __init__(self, num_heads, head_dim, num_clusters, ema_decay = 0.999, commitment = 1e-4):
        super().__init__()
        self.commitment = commitment
        self.ema_decay = ema_decay

        self.register_buffer('means', torch.randn(num_heads, num_clusters, head_dim))
        self.register_buffer('initted', torch.tensor(False))
        self.num_new_means = 0
        self.new_means = None

    @torch.no_grad()
    def init(self, x):
        if self.initted:
            return
        _, h, _, d, device, dtype = *x.shape, x.device, x.dtype

        num_clusters = self.means.shape[1]

        means = x.transpose(0, 1).contiguous().view(h, -1, d)
        num_samples = means.shape[1]

        if num_samples >= num_clusters:
            indices = torch.randperm(num_samples, device=device)[:num_clusters]
        else:
            indices = torch.randint(0, num_samples, (num_clusters,), device=device)

        means = means[:, indices]

        for _ in range(KMEAN_INIT_ITERS):
            means = kmeans_iter(x, means)

        self.num_new_means = 0
        self.means.data.copy_(means)
        self.initted.data.copy_(torch.tensor(True))

    @torch.no_grad()
    def update(self, new_means = None):
        new_means = default(new_means, self.new_means)
        assert exists(new_means), 'new kmeans has not been supplied'
        ema_inplace(self.means, new_means, self.ema_decay)

        del self.new_means
        self.new_means = None
        self.num_new_means = 0

    def forward(self, x, update_means = False):
        self.init(x)

        b, dtype = x.shape[0], x.dtype
        means = self.means.type(dtype)
        # x = F.normalize(x, 2, dim=-1).type(dtype)

        with torch.no_grad():
            dists, buckets = dists_and_buckets(x, means)

        routed_means = batched_index_select(expand_dim(means, 0, b), buckets)
        loss = F.mse_loss(x, routed_means) * self.commitment

        if update_means:
            with torch.no_grad():
                means = kmeans_iter(x, means, buckets)
            self.new_means = ema(self.new_means, means, self.num_new_means / (self.num_new_means + 1))
            self.num_new_means += 1

        return dists, buckets, loss

class DglAggregator(nn.Module):
    def __init__(self, dim, alpha):
        super(DglAggregator, self).__init__()
        self.dim = dim

        self.pi =  nn.Linear(self.dim, 1, bias=False)
        self.pj = nn.Linear(self.dim, 1, bias=False)
        self.q = nn.Linear(2*self.dim, self.dim, bias=False)
        self.r = nn.Linear(2*self.dim, self.dim, bias=False)
        self.M = nn.Linear(2*self.dim, 1, bias=False)
        self.Two = nn.Linear(2*self.dim, 2)

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.relu = nn.ReLU()
        self.dropout_local = nn.Dropout(config.dropout_local)
        self.dropout_attn = nn.Dropout(config.dropout_attn)
        self.norm = nn.LayerNorm(self.dim)
        # self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(config.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, h_v, h_d, h_p, h_t, g):
        with g.local_scope():
            ###item to item
            adj = g.edge_type_subgraph(['interacts'])
            adj.nodes['item'].data['ft'] = h_v
            adj.edges['interacts'].data['d'] = h_d
            # adj.apply_edges(fn.u_mul_v('ft','ft','e'), etype='interacts')
            adj.apply_edges(dis_u_mul_v, etype='interacts')
            e = self.pi(adj.edges['interacts'].data['e'])
            mask = torch.sigmoid(self.M(adj.edges['interacts'].data['mask']))
            e = e * mask
            # e = self.pi(adj.edges['interacts'].data['e'])
            # sim = adj.edges['interacts'].data['sim']
            # dis = adj.edges['interacts'].data['dis']
            # e = e.masked_fill((sim.unsqueeze(-1)<0.4) & (dis.unsqueeze(-1)>10), float('-inf'))
            # e= self.leakyrelu(e)

            
            # gumble = self.Two(adj.edges['interacts'].data['d_fea'])
            # gumble = F.gumbel_softmax(gumble, tau=0.1, hard=False)

            # adj.edges['interacts'].data['gumble'] = gumble
            # adj.update_all(udf_message, udf_edge_softmax, etype='interacts')

            adj.edges['interacts'].data['a'] = self.dropout_local(edge_softmax(adj['interacts'], e))
            adj.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'), etype='interacts')

            ###agg
            adj = g.edge_type_subgraph(['agg'])
            adj.nodes['target'].data['ft'] = h_t
            adj.edges['agg'].data['pos'] = h_p
            adj.apply_edges(fn.copy_src('ft','ft'))
            e = self.q(torch.cat([adj.edata['ft'], adj.edata['pos']], dim=-1))
            adj.edata['e'] = torch.tanh(e)

            last_nodes = adj.filter_nodes(lambda nodes: nodes.data['last']==1, ntype='item')
            last_feat = adj.nodes['item'].data['ft'][last_nodes]
            last_feat = last_feat.unsqueeze(1).repeat(1,config.order,1).view(-1, config.dim)
            # print(last_feat.shape)
            # last_feat = dgl.broadcast_nodes(adj, last_feat, ntype='target')
            # print(adj.nodes['target'].data['ft'].shape, last_feat.shape)
            f = self.r(torch.cat([adj.nodes['target'].data['ft'], last_feat], dim=-1))

            # adj.update_all(fn.copy_edge('ft', 'm'), fn.mean('m', 'mean'))
            # f = self.r(torch.cat([adj.nodes['target'].data['ft'], adj.nodes['target'].data['mean']], dim=-1))
            adj.nodes['target'].data['ft'] = f

            adj.update_all(udf_agg, fn.sum('m', 'ft'))

            return g.nodes['target'].data['ft']

# ignore weight decay for parameters in bias, batch norm and activation
def fix_weight_decay(model):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(map(lambda x: x in name, ['bias', 'batch_norm', 'activation'])):
            no_decay.append(param)
        else:
            decay.append(param)
    params = [{'params': decay}, {'params': no_decay, 'weight_decay': 0}]
    return params

class SessionGraph(nn.Module):
    def __init__(self, num_node):
        super(SessionGraph, self).__init__()
        self.num_node = num_node
        print(self.num_node)

        self.embedding = nn.Embedding(self.num_node, config.dim)
        self.pos_embedding = nn.Embedding(200, config.dim)
        self.len_embedding = nn.Embedding(200, config.dim)
        self.dis_embedding = nn.Embedding(200, config.dim)
        self.target_embedding = nn.Embedding(10, config.dim)
        self.group_embedding = nn.Embedding(1, config.dim)
        self.feat_drop = nn.Dropout(config.feat_drop)
        
        # Parameters        
        # self.kmeans = Kmeans(1, config.dim, 10, 0.999, 1e-4)
        self._handle = update_kmeans_on_backwards(self)

        self.local_agg = DglAggregator(config.dim, config.alpha)
        self.w_1 = nn.Parameter(torch.Tensor(2 * config.dim, config.dim))
        self.w_2 = nn.Parameter(torch.Tensor(config.dim, 1))
        self.glu1 = nn.Linear(config.dim, config.dim)
        self.glu2 = nn.Linear(config.dim, config.dim, bias=False)
        self.alpha    = nn.Parameter(torch.Tensor(config.order))
        self.phi = nn.Parameter(torch.Tensor(2))

        self.sc_sr = nn.ModuleList()
        for i in range(1):
            self.sc_sr.append(nn.Sequential(nn.Linear(config.dim, config.dim, bias=True),  nn.ReLU(), nn.Linear(config.dim, 2, bias=False), nn.Softmax(dim=-1)))

        # self.loss_function = nn.CrossEntropyLoss()
        self.loss_function = LabelSmoothSoftmaxCEV1(lb_smooth=config.lb_smooth, reduction='mean')
        print('weight_decay:', config.weight_decay)
        if config.weight_decay > 0:
            params = fix_weight_decay(self)
        else:
            params = self.parameters()
        self.optimizer = torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.lr_dc_step, gamma=config.lr_dc)

        self.reset_parameters()
        self.alpha.data = torch.zeros(config.order)
        self.alpha.data[0] = torch.tensor(1.0)
        self.phi.data[0] = 0
        self.phi.data[1] = 1

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(config.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        
    
    def forward(self, g, epoch=None, pos_mask=None, neg_mask=None, training=False):
        h_v = self.embedding(g.nodes['item'].data['iid'])
        h_v = self.feat_drop(h_v)
        h_v = F.normalize(h_v, dim=-1)
        
        h_d = self.dis_embedding(g.edges['interacts'].data['dis'])
        # dists, buckets, aux_loss = self.kmeans(h_v.unsqueeze(0).unsqueeze(0),  self.training)
        # buckets = buckets.squeeze(0).squeeze(0)

        # g.add_edges(*torch.where((buckets==buckets.view(-1,1))*mask), etype="interacts")

        # _, dists = dists.squeeze(0).squeeze(0).topk(3)
        # dists = torch.nn.functional.one_hot(dists, num_classes=10)
        # dists = torch.sum(dists.float(), dim=1)

        # with torch.no_grad():
        #     g.add_edges(*torch.where( ((h_v @ h_v.transpose(1,0)) * mask )>0), etype="interacts")

        h_p = self.pos_embedding(g.edges['agg'].data['pid'])
        h_r = self.target_embedding(g.nodes['target'].data['tid'])
        sr = self.local_agg(h_v, h_d, h_p, h_r, g)
        b = self.embedding.weight#[1:config.num_node]  # n_nodes x latent_size

        sr = F.normalize(sr, dim=-1)
        b = F.normalize(b, dim=-1)
        
        logits = torch.matmul(sr, b.transpose(1, 0))
        score = torch.softmax(12 * logits, dim=1)#.log()

        # if training:
        #     reg = torch.matmul(sr, sr.transpose(1, 0))
        #     reg.masked_fill(~((pos_mask+neg_mask).bool()), float('-inf'))
        #     reg_score = torch.softmax(12*reg, dim=1).log()
        #     reg_loss = -torch.sum(reg_score*pos_mask) / torch.sum(pos_mask)
        #     # print("sr", sr, "reg", reg_score)
        #     # exit()
        #     return score, reg_loss
        
        return score

        # phi = self.sc_sr[0](sr).unsqueeze(-1)
        phi = torch.softmax(self.phi, dim=-1)
        print(phi)
        phi = phi.unsqueeze(0).unsqueeze(-1).repeat(sr.size(0),1,1)
        mask = torch.zeros(phi.size(0), config.num_node).cuda()
        iids = torch.split(g.nodes['item'].data['iid'], g.batch_num_nodes('item').tolist())
        for i in range(len(mask)):
            mask[i, iids[i]] = 1
        # print(iids)
        logits_in = logits.masked_fill(~mask.bool(), float('-inf'))
        logits_ex = logits.masked_fill(mask.bool(), float('-inf'))
        score     = torch.softmax(12 * logits_in, dim=-1)
        score_ex  = torch.softmax(12 * logits_ex, dim=-1) 
        # if torch.isnan(score).any():
        #     score    = feat.masked_fill(score != score, 0)
        # if torch.isnan(score_ex).any():
        #     score_ex = score_ex.masked_fill(score_ex != score_ex, 0)
        assert not torch.isnan(score).any()
        assert not torch.isnan(score_ex).any()

        phi = phi.squeeze(1)
        # print(score.unsqueeze(1).shape, score_ex.unsqueeze(1).shape, phi.shape)
        score = (torch.cat((score.unsqueeze(1), score_ex.unsqueeze(1)), dim=1) * phi).sum(1)
        return torch.log(score)
    
    def agg(self, h_v, b, g):
        h_v = self.feat_drop(h_v)
        h_v = F.normalize(h_v, dim=-1)
        h_d = self.dis_embedding(g.edges['interacts'].data['dis'])
        h_p = self.pos_embedding(g.edges['agg'].data['pid'])
        h_r = self.target_embedding(g.nodes['target'].data['tid'])

        sr = self.local_agg(h_v, h_d, h_p, h_r, g)
        sr = F.normalize(sr, dim=-1)
        b = F.normalize(b, dim=-1)
        
        logits = torch.matmul(sr, b.transpose(1, 0))
        score = torch.softmax(12 * logits, dim=1)#.log()
        
        return score


class DglAggregator2(nn.Module):
    def __init__(self, dim, alpha):
        super(DglAggregator2, self).__init__()
        self.dim = dim

        self.pi =  nn.Linear(self.dim, 1, bias=False)
        self.pj = nn.Linear(self.dim, 1, bias=False)
        self.q = nn.Linear(2*self.dim, self.dim, bias=False)
        self.r = nn.Linear(2*self.dim, self.dim, bias=False)
        self.M = nn.Linear(2*self.dim, 1, bias=False)
        self.Two = nn.Linear(2*self.dim, 2)

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.relu = nn.ReLU()
        self.dropout_local = nn.Dropout(config.dropout_local)
        self.dropout_attn = nn.Dropout(config.dropout_attn)
        self.norm = nn.LayerNorm(self.dim)
        # self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(config.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, h_v, h_d, h_p, h_t, g):
        with g.local_scope():
            ###item to item
            adj = g.edge_type_subgraph(['interacts'])
            adj.nodes['item'].data['ft'] = h_v
            adj.edges['interacts'].data['d'] = h_d
            # adj.apply_edges(fn.u_mul_v('ft','ft','e'), etype='interacts')
            adj.apply_edges(dis_u_mul_v, etype='interacts')
            e = self.pi(adj.edges['interacts'].data['e'])
            mask = torch.sigmoid(self.M(adj.edges['interacts'].data['mask']))
            e = e * mask

            adj.edges['interacts'].data['a'] = self.dropout_local(edge_softmax(adj['interacts'], e))
            adj.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'), etype='interacts')

            ###agg
            adj = g.edge_type_subgraph(['agg'])
            adj.nodes['target'].data['ft'] = h_t
            adj.edges['agg'].data['pos'] = h_p
            adj.apply_edges(fn.copy_src('ft','ft'))
            e = self.q(torch.cat([adj.edata['ft'], adj.edata['pos']], dim=-1))
            adj.edata['e'] = torch.tanh(e)

            last_nodes = adj.filter_nodes(lambda nodes: nodes.data['last']==1, ntype='item')
            last_feat = adj.nodes['item'].data['ft'][last_nodes]
            last_feat = last_feat.unsqueeze(1).repeat(1,config.order,1).view(-1, config.dim)
            f = self.r(torch.cat([adj.nodes['target'].data['ft'], last_feat], dim=-1))
            adj.nodes['target'].data['ft'] = f

            adj.update_all(udf_agg, fn.sum('m', 'ft'))

            return g.nodes['target'].data['ft']

class SessionGraph2(nn.Module):
    def __init__(self, num_node):
        super(SessionGraph2, self).__init__()
        self.num_node = num_node
        print(self.num_node)

        self.embedding = nn.Embedding(self.num_node, config.dim)
        self.pos_embedding = nn.Embedding(200, config.dim)
        self.len_embedding = nn.Embedding(200, config.dim)
        self.dis_embedding = nn.Embedding(200, config.dim)
        self.target_embedding = nn.Embedding(10, config.dim)
        self.group_embedding = nn.Embedding(1, config.dim)
        self.feat_drop = nn.Dropout(config.feat_drop*3)
        
        # Parameters        
        # self.kmeans = Kmeans(1, config.dim, 10, 0.999, 1e-4)
        self._handle = update_kmeans_on_backwards(self)

        self.local_agg = DglAggregator2(config.dim, config.alpha)
        self.w_1 = nn.Parameter(torch.Tensor(2 * config.dim, config.dim))
        self.w_2 = nn.Parameter(torch.Tensor(config.dim, 1))
        self.glu1 = nn.Linear(config.dim, config.dim)
        self.glu2 = nn.Linear(config.dim, config.dim, bias=False)
        self.alpha    = nn.Parameter(torch.Tensor(config.order))
        self.phi = nn.Parameter(torch.Tensor(2))

        self.sc_sr = nn.ModuleList()
        for i in range(1):
            self.sc_sr.append(nn.Sequential(nn.Linear(config.dim, config.dim, bias=True),  nn.ReLU(), nn.Linear(config.dim, 2, bias=False), nn.Softmax(dim=-1)))

        # self.loss_function = nn.CrossEntropyLoss()
        self.loss_function = LabelSmoothSoftmaxCEV1(lb_smooth=config.lb_smooth, reduction='mean')
        print('weight_decay:', config.weight_decay)
        if config.weight_decay > 0:
            params = fix_weight_decay(self)
        else:
            params = self.parameters()
        self.optimizer = torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.lr_dc_step, gamma=config.lr_dc)

        self.reset_parameters()
        self.alpha.data = torch.zeros(config.order)
        self.alpha.data[0] = torch.tensor(1.0)
        self.phi.data[0] = 0
        self.phi.data[1] = 1

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(config.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        
    
    def forward(self, g, epoch=None, pos_mask=None, neg_mask=None, training=False):
        h_v = self.embedding(g.nodes['item'].data['iid'])
        h_v = self.feat_drop(h_v)
        h_v = F.normalize(h_v, dim=-1)
        
        h_d = self.dis_embedding(g.edges['interacts'].data['dis'])

        h_p = self.pos_embedding(g.edges['agg'].data['pid'])
        h_r = self.target_embedding(g.nodes['target'].data['tid'])
        sr = self.local_agg(h_v, h_d, h_p, h_r, g)
        b = self.embedding.weight#[1:config.num_node]  # n_nodes x latent_size

        sr = F.normalize(sr, dim=-1)
        b = F.normalize(b, dim=-1)
        
        logits = torch.matmul(sr, b.transpose(1, 0))
        score = torch.softmax(12 * logits, dim=1)#.log()
        
        return score
    
    def agg(self, h_v, b, g):
        h_v = self.feat_drop(h_v)
        h_v = F.normalize(h_v, dim=-1)
        h_d = self.dis_embedding(g.edges['interacts'].data['dis'])
        h_p = self.pos_embedding(g.edges['agg'].data['pid'])
        h_r = self.target_embedding(g.nodes['target'].data['tid'])

        sr = self.local_agg(h_v, h_d, h_p, h_r, g)
        sr = F.normalize(sr, dim=-1)
        b = F.normalize(b, dim=-1)
        
        logits = torch.matmul(sr, b.transpose(1, 0))
        score = torch.softmax(12 * logits, dim=1)#.log()
        
        return score


       

class Ensamble(nn.Module):
    def __init__(self, num_node):
        super(Ensamble, self).__init__()
        # self.embedding = nn.Embedding(num_node, config.dim)
        # self.mix_weights = nn.Linear(2*config.dim, 2)

        self.model1 = SessionGraph(num_node = num_node)
        self.model2 = SessionGraph2(num_node = num_node)
        self.phi = nn.Parameter(torch.Tensor(2))
        self.loss = LabelSmoothSoftmaxCEV1(lb_smooth=config.lb_smooth, reduction='mean')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.lr_dc_step, gamma=config.lr_dc)

        self.reset_parameters()
        self.phi.data[0] = 0
        self.phi.data[1] = 0

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(config.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def forward(self, g, epoch=None, pos_mask=None, neg_mask=None, training=False):
        

        # with g.local_scope():
        #     adj = g.edge_type_subgraph(['agg'])
        #     adj.nodes['item'].data['ft1'] = self.model1.embedding(g.nodes['item'].data['iid'])
        #     adj.nodes['item'].data['ft2'] = self.model2.embedding(g.nodes['item'].data['iid'])
        #     # last_nodes = adj.filter_nodes(lambda nodes: nodes.data['last']==1, ntype='item')
        #     # mean1 = adj.nodes['item'].data['ft1'][last_nodes]
        #     # mean2 = adj.nodes['item'].data['ft2'][last_nodes]
        #     adj.update_all(fn.copy_src('ft1','ft1'), fn.mean('ft1', 'mean1'))
        #     adj.update_all(fn.copy_src('ft2','ft2'), fn.mean('ft2', 'mean2'))
        #     mean1 = adj.nodes['target'].data['mean1']
        #     mean2 = adj.nodes['target'].data['mean2']
        #     phi = self.mix_weights(torch.cat([mean1, mean2], dim=-1))
        #     phi = torch.softmax(phi, dim=-1)
        #     phi = phi.unsqueeze(-1)

        scores1 = self.model1(g, epoch, pos_mask, neg_mask, training)
        scores2 = self.model2(g, epoch, pos_mask, neg_mask, training)

        phi = torch.softmax(self.phi, dim=-1)
        # print(phi)
        phi = phi.unsqueeze(0).unsqueeze(-1).repeat(scores1.size(0),1,1)
        score_mix = torch.sum(torch.cat([scores1.unsqueeze(1), scores2.unsqueeze(1)], dim=1) * phi, dim=1)

        return [scores1, scores2, score_mix]

    def loss_function(self, scores, targets):
        scores1, scores2, score_mix = scores
        score_cat = torch.cat([scores1.unsqueeze(1), scores2.unsqueeze(1)], dim=1)
        loss1 = self.model1.loss_function(scores1.log(), targets)
        loss2 = self.model2.loss_function(scores2.log(), targets)
        loss_mix = self.loss(score_mix.log(), targets)

        epsilon = 1e-8
        score_mix = torch.unsqueeze(score_mix, 1)
        kl_loss = torch.sum(score_mix * (torch.log(score_mix + epsilon) - torch.log(score_cat + epsilon)), dim=-1)
        regularization_loss  = torch.mean(torch.sum(kl_loss, dim=-1), dim=-1)
        return [loss1, loss2, loss_mix,  regularization_loss]
