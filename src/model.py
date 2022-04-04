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



class GATLayer(nn.Module):
    def __init__(self, dim):
        super(GATLayer, self).__init__()
        self.dim = dim

        self.pi =  nn.Linear(self.dim, 1, bias=False)
        self.M = nn.Linear(2*self.dim, 1, bias=False)
        self.dropout_local = nn.Dropout(config.dropout_local)

    def forward(self, h_v, h_d, g):
        with g.local_scope():
            ###item to item
            adj = g.edge_type_subgraph(['interacts'])
            adj.nodes['item'].data['ft'] = h_v
            adj.edges['interacts'].data['d'] = h_d
            adj.apply_edges(dis_u_mul_v, etype='interacts')
            e = self.pi(adj.edges['interacts'].data['e'])
            mask = torch.sigmoid(self.M(adj.edges['interacts'].data['mask']))
            e = e * mask

            adj.edges['interacts'].data['a'] = self.dropout_local(edge_softmax(adj['interacts'], e))
            adj.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'), etype='interacts')
            return adj.nodes['item'].data['ft']

class GAT(nn.Module):
    def __init__(self, dim):
        super(GAT, self).__init__()
        self.layer1 = GATLayer(dim)
        # self.layer2 = GATLayer(dim)

    def forward(self, h_0, h_d, g):
        h1 = self.layer1(h_0, h_d, g)
        return h1
        h1 = F.elu(h1+h_0)
        h2 = self.layer2(h1, h_d, g)
        h2 = F.elu(h2+h1)
        return h2


class PosAggregator(nn.Module):
    def __init__(self, dim):
        super(PosAggregator, self).__init__()
        self.dim = dim

        self.q = nn.Linear(2*self.dim, self.dim, bias=False)
        self.r = nn.Linear(2*self.dim, self.dim, bias=False)


    def forward(self, h_v, h_p, h_t, g):
        with g.local_scope():
            adj = g.edge_type_subgraph(['agg'])
            adj.nodes['item'].data['ft'] = h_v
            adj.nodes['target'].data['ft'] = h_t
            adj.edges['agg'].data['pos'] = h_p
            adj.apply_edges(fn.copy_src('ft','ft'))
            e = self.q(torch.cat([adj.edata['ft'], adj.edata['pos']], dim=-1))
            adj.edata['e'] = torch.tanh(e)

            last_nodes = adj.filter_nodes(lambda nodes: nodes.data['last']==1, ntype='item')
            last_feat = adj.nodes['item'].data['ft'][last_nodes]
            last_feat = last_feat.unsqueeze(1).repeat(1,1,1).view(-1, config.dim)

            f = self.r(torch.cat([adj.nodes['target'].data['ft'], last_feat], dim=-1))
            adj.nodes['target'].data['ft'] = f
            adj.update_all(udf_agg, fn.sum('m', 'ft'))

            return g.nodes['target'].data['ft']

class SessionGraph4(nn.Module):
    def __init__(self, num_node, feat_drop=config.feat_drop):
        super(SessionGraph4, self).__init__()
        self.num_node = num_node
        print(self.num_node)

        self.embedding = nn.Embedding(self.num_node, config.dim)
        self.pos_embedding = nn.Embedding(200, config.dim)
        self.dis_embedding = nn.Embedding(50, config.dim)
        self.target_embedding = nn.Embedding(10, config.dim)
        self.feat_drop = nn.Dropout(feat_drop)
        # self.phi = nn.Parameter(torch.Tensor(2))
        self.sc_sr = nn.Sequential(nn.Linear(config.dim, config.dim, bias=True),  nn.ReLU(), nn.Linear(config.dim, 2, bias=False), nn.Softmax(dim=-1))

        self.gat = GAT(config.dim)
        self.agg = PosAggregator(config.dim)

        self.loss_function = LabelSmoothSoftmaxCEV1(lb_smooth=config.lb_smooth, reduction='mean')
        print('weight_decay:', config.weight_decay)
        if config.weight_decay > 0:
            params = fix_weight_decay(self)
        else:
            params = self.parameters()
        self.optimizer = torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.lr_dc_step, gamma=config.lr_dc)

        self.reset_parameters()

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

        h = self.gat(h_v, h_d, g)
        sr = self.agg(h, h_p, h_r, g)    

        b = self.embedding.weight#[1:config.num_node]  # n_nodes x latent_size

        sr = F.normalize(sr, dim=-1)
        if training:
            return sr

        b = F.normalize(b, dim=-1)
        
        logits = torch.matmul(sr, b.transpose(1, 0))
        # score = torch.softmax(12 * logits, dim=1).log()
        # return score

        # phi = torch.softmax(self.phi, dim=-1)
        # phi = phi.unsqueeze(0).unsqueeze(-1).repeat(sr.size(0),1,1)
        phi = self.sc_sr(sr).unsqueeze(-1)
        mask = torch.zeros(phi.size(0), config.num_node).cuda()
        iids = torch.split(g.nodes['item'].data['iid'], g.batch_num_nodes('item').tolist())
        for i in range(len(mask)):
            mask[i, iids[i]] = 1
        # print(iids)
        logits_in = logits.masked_fill(~mask.bool(), float('-inf'))
        logits_ex = logits.masked_fill(mask.bool(), float('-inf'))
        score     = torch.softmax(12 * logits_in, dim=-1)
        score_ex  = torch.softmax(12 * logits_ex, dim=-1) 
        assert not torch.isnan(score).any()
        assert not torch.isnan(score_ex).any()

        phi = phi.squeeze(1)
        score = (torch.cat((score.unsqueeze(1), score_ex.unsqueeze(1)), dim=1) * phi).sum(1)
        return torch.log(score)

    def get_score(self, sr, g=None):
        b = self.embedding.weight#[1:config.num_node]  # n_nodes x latent_size
        b = F.normalize(b, dim=-1)
        
        logits = torch.matmul(sr, b.transpose(1, 0))
        # score = torch.softmax(12 * logits, dim=1).log()
        # return score

        # phi = torch.softmax(self.phi, dim=-1)
        # phi = phi.unsqueeze(0).unsqueeze(-1).repeat(sr.size(0),1,1)
        phi = self.sc_sr(sr).unsqueeze(-1)
        mask = torch.zeros(phi.size(0), config.num_node).cuda()
        iids = torch.split(g.nodes['item'].data['iid'], g.batch_num_nodes('item').tolist())
        for i in range(len(mask)):
            mask[i, iids[i]] = 1
        # print(iids)
        logits_in = logits.masked_fill(~mask.bool(), float('-inf'))
        logits_ex = logits.masked_fill(mask.bool(), float('-inf'))
        score     = torch.softmax(12 * logits_in, dim=-1)
        score_ex  = torch.softmax(12 * logits_ex, dim=-1) 
        assert not torch.isnan(score).any()
        assert not torch.isnan(score_ex).any()

        phi = phi.squeeze(1)
        score = (torch.cat((score.unsqueeze(1), score_ex.unsqueeze(1)), dim=1) * phi).sum(1)
        return score
        return torch.log(score)
    

class Ensamble(nn.Module):
    def __init__(self, num_node):
        super(Ensamble, self).__init__()

        self.model1 = SessionGraph4(num_node = num_node, feat_drop=config.feat_drop)
        self.model2 = SessionGraph4(num_node = num_node, feat_drop=config.feat_drop*2)
        self.model3 = SessionGraph4(num_node = num_node, feat_drop=config.feat_drop)
        self.phi = nn.Parameter(torch.Tensor(3))
        self.loss_function = LabelSmoothSoftmaxCEV1(lb_smooth=config.lb_smooth, reduction='mean')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.lr_dc_step, gamma=config.lr_dc)

        self.reset_parameters()
        self.phi.data[0] = 0
        self.phi.data[1] = 0
        self.phi.data[2] = 0

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(config.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def forward(self, g, targets, epoch=None, training=False):
        sr1 = self.model1(g, epoch, training=True)
        sr2 = self.model2(g, epoch, training=True)
        sr3 = self.model3(g, epoch, training=True)
        scores1 = self.model1.get_score(sr1, g)
        scores2 = self.model2.get_score(sr2, g)
        scores3 = self.model3.get_score(sr3, g)

        phi = torch.softmax(self.phi, dim=-1)
        phi = phi.unsqueeze(0).unsqueeze(-1).repeat(scores1.size(0),1,1)
        score_mix = torch.sum(torch.cat([scores1.unsqueeze(1), scores2.unsqueeze(1), scores3.unsqueeze(1)], dim=1) * phi, dim=1)

        if not training:
            return score_mix

        loss = self.loss_function(score_mix.log(), targets)
        return loss

        score_cat = torch.cat([scores1.unsqueeze(1), scores2.unsqueeze(1)], dim=1)
        loss1 = self.model1.loss_function(scores1.log(), targets) 
        loss2 = self.model2.loss_function(scores2.log(), targets) 
        loss3 = self.model3.loss_function(scores3.log(), targets) 

        epsilon = 1e-8
        score_mix = torch.unsqueeze(score_mix, 1)
        with torch.no_grad():
            kl_loss = torch.sum(score_mix * (torch.log(score_mix + epsilon) - torch.log(score_cat + epsilon)), dim=-1)
        regularization_loss  = torch.mean(torch.sum(kl_loss, dim=-1), dim=-1)

        return loss1, loss2, regularization_loss
