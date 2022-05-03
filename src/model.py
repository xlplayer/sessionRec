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
from dgl.base import DGLError
from dgl.utils import expand_as_pair
from dgl.nn.pytorch.utils import Identity
import dgl.nn.pytorch as dglnn
from entmax import sparsemax, entmax15, entmax_bisect
from label_smooth import LabelSmoothSoftmaxCEV1, FocalLoss


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
    def __init__(self, dim, num_heads, idx):
        super(GATLayer, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.idx = idx

        self.pi =  nn.Linear(self.dim * self.num_heads, 1, bias=False)
        self.fc = nn.Linear(self.dim, self.dim*self.num_heads)
        self.feat_drop = nn.Dropout(config.feat_drop)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, h_v, g):
        with g.local_scope():
            ###item to item
            # h_v = self.feat_drop(h_v)
            adj = g.edge_type_subgraph(['interacts'+str(self.idx)])
            adj.nodes['item'].data['ft'] = self.fc(h_v)
            adj.apply_edges(fn.u_mul_v('ft','ft','e'), etype='interacts'+str(self.idx))
            e = self.pi(adj.edges['interacts'+str(self.idx)].data['e'])
            e = self.leaky_relu(e)

            adj.edges['interacts'+str(self.idx)].data['a'] = edge_softmax(adj['interacts'+str(self.idx)], e)
            adj.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'), etype='interacts'+str(self.idx))
            # return adj.nodes['item'].data['ft']
            rst = adj.nodes['item'].data['ft']
            return torch.max(rst.view(-1,self.num_heads,config.dim), dim=1)[0]

class GAT(nn.Module):
    def __init__(self, dim, num_heads=8, idx=0):
        super(GAT, self).__init__()
        self.layer1 = GATLayer(dim, num_heads = num_heads, idx=idx)
        # self.layer2 = GATLayer(dim)

    def forward(self, h_0, g):
        h1 = self.layer1(h_0, g)
        # return h1
        return h_0+h1
       


# pylint: enable=W0235
class GATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_d = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_d, gain=gain)
        nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(-1, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            # ed = (d_feat.view(-1, self._num_heads, self._out_feats) *self.attn_d).sum(dim=-1).unsqueeze(-1)
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(1, -1, self._out_feats)
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst


class PosAggregator(nn.Module):
    def __init__(self, dim, last_L=1):
        super(PosAggregator, self).__init__()
        self.dim = dim

        self.q = nn.Linear(2*self.dim, self.dim, bias=False)
        self.r = nn.Linear(2*self.dim, self.dim, bias=False)
        self.GRU = nn.GRU(self.dim, self.dim, 1, True, True)
        self.last_L = last_L


    def forward(self, h_v, h_p, h_t, g):
        with g.local_scope():
            adj = g.edge_type_subgraph(['agg'])
            adj.nodes['item'].data['ft'] = h_v
            adj.nodes['target'].data['ft'] = h_t
            adj.edges['agg'].data['pos'] = h_p
            adj.apply_edges(fn.copy_src('ft','ft'))
            e = self.q(torch.cat([adj.edata['ft'], adj.edata['pos']], dim=-1))
            adj.edata['e'] = torch.tanh(e)

            last_nodes = adj.filter_nodes(lambda nodes: nodes.data['last'][:,0]==1, ntype='item')
            last_feat = adj.nodes['item'].data['ft'][last_nodes]
            last_feat = last_feat.unsqueeze(1).repeat(1,1,1).view(-1, config.dim)

            # f = self.r(torch.cat([adj.nodes['target'].data['ft'], last_feat], dim=-1))
            # adj.nodes['target'].data['ft'] = f
            adj.update_all(udf_agg, fn.sum('m', 'ft'))

            return g.nodes['target'].data['ft'], last_feat


class HardSession(nn.Module):
    def __init__(self, num_node, feat_drop=config.feat_drop, num_heads=8, order=3, pos_embedding=None, target_embedding = None, mask=True, share=True):
        super(HardSession, self).__init__()
        self.num_node = num_node
        self.order = order
        self.mask = mask
        print(self.num_node)
        self.embedding = nn.Embedding(self.num_node, config.dim)

        if share:
            self.pos_embedding = pos_embedding
            self.target_embedding = target_embedding
        else:
            self.pos_embedding = nn.Embedding(200, config.dim)
            self.target_embedding = nn.Embedding(10, config.dim)
        self.feat_drop = nn.Dropout(feat_drop)
        
        self.gat1   = nn.ModuleList()
        self.gat2 = nn.ModuleList()
        self.agg = nn.ModuleList()
        self.fc_sr = nn.ModuleList()
        self.sc_sr = nn.Sequential(nn.Linear(config.dim, config.dim, bias=True),  nn.ReLU(), nn.Linear(config.dim, 2, bias=False), nn.Softmax(dim=-1))
        for i in range(self.order):
            self.gat1.append(dglnn.HeteroGraphConv({"interacts"+str(i):GATConv(config.dim, config.dim, num_heads, feat_drop, feat_drop, residual=True, allow_zero_in_degree=True)}, aggregate='sum'))
            self.gat2.append(dglnn.HeteroGraphConv({"interacts"+str(i):GATConv(config.dim, config.dim, num_heads, feat_drop, feat_drop, residual=True, allow_zero_in_degree=True)}, aggregate='sum'))
            self.agg.append(PosAggregator(config.dim, i+1))
            self.fc_sr.append(nn.Linear(2*config.dim, config.dim, bias=False))
                    

        # self.alpha = nn.Parameter(torch.Tensor(self.order))
        self.register_buffer('alpha', torch.Tensor(self.order))
        self.loss_function = LabelSmoothSoftmaxCEV1(lb_smooth=config.lb_smooth, reduction='mean')
        # self.loss_function = FocalLoss(gamma=2)
        print('weight_decay:', config.weight_decay)
        if config.weight_decay > 0:
            params = fix_weight_decay(self)
        else:
            params = self.parameters()
        self.optimizer = torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.lr_dc_step, gamma=config.lr_dc)

        self.reset_parameters()
        self.alpha.data = torch.zeros(self.order)
        # self.alpha.data[0] = torch.tensor(1.0)
        
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(config.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def forward(self, g, targets, epoch=None, training=False):
        h_v = self.embedding(g.nodes['item'].data['iid'])
        h_v = self.feat_drop(h_v)
        h_v = F.normalize(h_v, dim=-1)

        h_p = self.pos_embedding(g.edges['agg'].data['pid'])
        h_r = self.target_embedding(g.nodes['target'].data['tid'])

        feat, last_feat = [],[]
        for i in range(self.order):
            h1 = self.gat1[i](g.edge_type_subgraph(['interacts'+str(i)]), {'item':h_v})
            h1 = torch.max(h1['item'], dim=1)[0]
            h2 = self.gat2[i](g.reverse(copy_edata=True).edge_type_subgraph(['interacts'+str(i)]), {'item':h_v})
            h2 = torch.max(h2['item'], dim=1)[0]
            h = h1+h2
            h = F.normalize(h, dim=-1)
            x, y = self.agg[i](h, h_p, h_r, g)
            feat.append(x.unsqueeze(1))
            last_feat.append(y.unsqueeze(1))
        
        sr_g = torch.cat(feat, dim=1)
        sr_l = torch.cat(last_feat, dim=1)
        sr   = torch.cat([sr_l, sr_g], dim=-1)
        sr   = torch.cat([self.fc_sr[i](sr).unsqueeze(1) for i, sr in enumerate(torch.unbind(sr, dim=1))], dim=1)

        sr = F.normalize(sr, dim=-1)
        b = self.embedding.weight
        b = F.normalize(b, dim=-1)

        logits = sr @ b.t()

        if self.mask:
            phi = self.sc_sr(sr).unsqueeze(-1)
            mask = torch.zeros(phi.size(0), config.num_node).cuda()
            iids = torch.split(g.nodes['item'].data['iid'], g.batch_num_nodes('item').tolist())
            for i in range(len(mask)):
                mask[i, iids[i]] = 1

            logits_in = logits.masked_fill(~mask.bool().unsqueeze(1), float('-inf'))
            logits_ex = logits.masked_fill(mask.bool().unsqueeze(1), float('-inf'))
            score     = torch.softmax(12 * logits_in.squeeze(), dim=-1)
            score_ex  = torch.softmax(12 * logits_ex.squeeze(), dim=-1)
            if self.order == 1:
                phi = phi.squeeze(1)
                score = (torch.cat((score.unsqueeze(1), score_ex.unsqueeze(1)), dim=1) * phi).sum(1)
            else:
                score = (torch.cat((score.unsqueeze(2), score_ex.unsqueeze(2)), dim=2) * phi).sum(2)

        else:
            score = torch.softmax(12 * logits.squeeze(), dim=-1)


        if self.order>1:
            alpha = torch.softmax(self.alpha.unsqueeze(0), dim=-1).view(1, self.alpha.size(0), 1)
            score = (score * alpha.repeat(score.size(0), 1, 1)).sum(1)
        else:
            score = score.squeeze(1)

        score = torch.log(score)

        if not training:
            return score

        loss = self.loss_function(score, targets)
        return loss
        
class EasySession(nn.Module):
    def __init__(self, num_node, feat_drop=config.feat_drop, num_heads=8, order=3, pos_embedding=None, target_embedding = None, mask=False, share=True):
        super(EasySession, self).__init__()
        self.num_node = num_node
        self.order = order
        self.mask = mask
        print(self.num_node)
        self.register_buffer('alpha', torch.Tensor(self.order))
        self.embedding = nn.Embedding(self.num_node, config.dim)

        if share:
            self.pos_embedding = pos_embedding
            self.target_embedding = target_embedding
        else:
            self.pos_embedding = nn.Embedding(200, config.dim)
            self.target_embedding = nn.Embedding(10, config.dim)
        self.feat_drop = nn.Dropout(feat_drop)
        self.gat1   = nn.ModuleList()
        self.gat2 = nn.ModuleList()
        self.agg = nn.ModuleList()
        self.fc_sr = nn.ModuleList()
        self.sc_sr = nn.Sequential(nn.Linear(config.dim, config.dim, bias=True),  nn.ReLU(), nn.Linear(config.dim, 2, bias=False), nn.Softmax(dim=-1))
        for i in range(self.order):
            self.gat1.append(GAT(config.dim, num_heads=num_heads, idx=i))
            self.gat2.append(GAT(config.dim, num_heads=num_heads, idx=i))
            self.agg.append(PosAggregator(config.dim))
            self.fc_sr.append(nn.Linear(2*config.dim, config.dim, bias=False))

        self.loss_function = LabelSmoothSoftmaxCEV1(lb_smooth=config.lb_smooth, reduction='mean')
        print('weight_decay:', config.weight_decay)
        if config.weight_decay > 0:
            params = fix_weight_decay(self)
        else:
            params = self.parameters()
        self.optimizer = torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.lr_dc_step, gamma=config.lr_dc)

        self.reset_parameters()
        self.alpha.data = torch.zeros(self.order)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(config.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        
    
    def forward(self, g, targets, epoch=None, training=False):
        h_v = self.embedding(g.nodes['item'].data['iid'])
        h_v = self.feat_drop(h_v)
        h_v = F.normalize(h_v, dim=-1)
        
        h_p = self.pos_embedding(g.edges['agg'].data['pid'])
        h_r = self.target_embedding(g.nodes['target'].data['tid'])

        feat, last_feat = [],[]
        for i in range(self.order):
            h1 =self.gat1[i](h_v, g)
            h2 = self.gat2[i](h_v, g.reverse(copy_edata=True).edge_type_subgraph(['interacts'+str(i)]))
            h = h1+h2
            h = F.normalize(h, dim=-1)

            x, y = self.agg[i](h, h_p, h_r, g)
            feat.append(x.unsqueeze(1))
            last_feat.append(y.unsqueeze(1))

       
        sr_g = torch.cat(feat, dim=1)
        sr_l = torch.cat(last_feat, dim=1)
        sr   = torch.cat([sr_l, sr_g], dim=-1)
        sr   = torch.cat([self.fc_sr[i](sr).unsqueeze(1) for i, sr in enumerate(torch.unbind(sr, dim=1))], dim=1)

        sr = F.normalize(sr, dim=-1)
        b = self.embedding.weight
        b = F.normalize(b, dim=-1)

        logits = sr @ b.t()

        if self.mask:
            phi = self.sc_sr(sr).unsqueeze(-1)
            mask = torch.zeros(phi.size(0), config.num_node).cuda()
            iids = torch.split(g.nodes['item'].data['iid'], g.batch_num_nodes('item').tolist())
            for i in range(len(mask)):
                mask[i, iids[i]] = 1

            logits_in = logits.masked_fill(~mask.bool().unsqueeze(1), float('-inf'))
            logits_ex = logits.masked_fill(mask.bool().unsqueeze(1), float('-inf'))
            score     = torch.softmax(12 * logits_in.squeeze(), dim=-1)
            score_ex  = torch.softmax(12 * logits_ex.squeeze(), dim=-1)
            if self.order == 1:
                phi = phi.squeeze(1)
                score = (torch.cat((score.unsqueeze(1), score_ex.unsqueeze(1)), dim=1) * phi).sum(1)
            else:
                score = (torch.cat((score.unsqueeze(2), score_ex.unsqueeze(2)), dim=2) * phi).sum(2)

        else:
            score = torch.softmax(12 * logits.squeeze(), dim=-1)
            

        if self.order>1:
            alpha = torch.softmax(self.alpha.unsqueeze(0), dim=-1).view(1, self.alpha.size(0), 1)
            score = (score * alpha.repeat(score.size(0), 1, 1)).sum(1)
        else:
            score = score.squeeze(1)

        score = torch.log(score)

        if not training:
            return score

        loss = self.loss_function(score, targets)
        return loss


class Ensamble(nn.Module):
    def __init__(self, num_node, feat_drop=config.feat_drop, num_heads=8, order=3):
        super(Ensamble, self).__init__()
        self.num_node = num_node
        self.order = order

        self.pos_embedding = nn.Embedding(200, config.dim)
        self.target_embedding = nn.Embedding(10, config.dim)
        

        self.esay = EasySession(num_node=num_node, feat_drop=feat_drop, num_heads=num_heads, order=order, pos_embedding=self.pos_embedding, target_embedding = self.target_embedding, mask=False, share=True)
        self.hard = HardSession(num_node=num_node, feat_drop=feat_drop, num_heads=num_heads, order=order, pos_embedding=self.pos_embedding, target_embedding = self.target_embedding, mask=True, share=True)
        
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
        
    
    def forward(self, g, targets, epoch=None, training=False):
        score1 = self.esay(g, targets, training=False)
        score2 = self.hard(g, targets, training=False)
        score = score1+score2

        if not training:
            return score

        loss = self.loss_function(score, targets)
        return loss