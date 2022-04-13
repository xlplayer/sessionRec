from collections import defaultdict
from turtle import pos
from numpy.lib.twodim_base import mask_indices
import config
import time
import datetime
import torch
import numpy as np
import random
import pickle
from tqdm import tqdm
from model import Ensamble,Ensamble2, SessionGraph4
from utils import Data
import torch.nn.functional as F
import networkx as nx
import scipy.sparse as sp
from sklearn.cluster import KMeans
import dgl


def init_seed(seed=None):
    dgl.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

def collate_fn_mix(samples):
    g0, g1, target0, target1 = zip(*samples)
    g0 = dgl.batch(g0)
    g1 = dgl.batch(g1)
    return g0, g1, torch.tensor(target0), torch.tensor(target1)

def collate_fn(samples):
    g, target = zip(*samples)
    g = dgl.batch(g)
    return g, torch.tensor(target)


def train_test(model1, model2, train_data, test_data, epoch, train_sessions):
    print('start training: ', datetime.datetime.now())
    model1.train()
    model2.train()

    total_loss = 0.0
    if config.mixup:
        train_loader = torch.utils.data.DataLoader(train_data, num_workers=20, batch_size=config.batch_size,
                                               shuffle=False, pin_memory=True, collate_fn=collate_fn_mix)
    else:
        train_loader = torch.utils.data.DataLoader(train_data, num_workers=20, batch_size=config.batch_size,
                                               shuffle=False, pin_memory=True, collate_fn=collate_fn)
    
    with tqdm(train_loader) as t:
        for data in t:
            # torch.cuda.empty_cache()
            model1.optimizer.zero_grad()
            model2.optimizer.zero_grad()
            if config.mixup:
                g0, g1, target0, target1 = data
                g0 = g0.to(torch.device('cuda'))
                g1 = g1.to(torch.device('cuda'))
                targets0 = trans_to_cuda(target0).long()
                targets1 = trans_to_cuda(target1).long()
                score0 = model1(g0, epoch, training=False)
                score1 = model1(g1, epoch, training=False)

                # l = config.l
                l = 1-(epoch/10)**2
                score = l * score0 + (1-l) * score1
                loss1 = l * model1.loss_function(score, targets0) + (1 - l) *  model1.loss_function(score, targets1)
                loss2 = model2(g0, targets0, training=True)
                loss = loss1+loss2
            else:
                g, target = data
                g = g.to(torch.device('cuda'))
                targets = trans_to_cuda(target).long()
                loss1 = model1(g, targets, training=True)
                loss2 = model2(g, targets, training=True)
                loss = loss1 + loss2
            


            t.set_postfix(
                        loss = loss.item(),
                        lr = model1.optimizer.state_dict()['param_groups'][0]['lr'])
            loss.backward()
            model1.optimizer.step()
            model2.optimizer.step()
            total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)

    model1.scheduler.step()
    model2.scheduler.step()

    print('start predicting: ', datetime.datetime.now())
    model1.eval()
    model2.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        test_loader = torch.utils.data.DataLoader(test_data, num_workers=20, batch_size=config.batch_size,
                                                shuffle=True, pin_memory=True, collate_fn=collate_fn)
        result = []
        hit20, mrr20 = [], []
        hit10, mrr10 = [], []
        hit5, mrr5 = [], []
        for data in test_loader:
            g, target = data
            g = g.to(torch.device('cuda'))
            targets = trans_to_cuda(target).long()
            scores =  model1(g, epoch) + model2(g, epoch)

            # scores = scores[-1] ###mix score
            assert not torch.isnan(scores).any()

            sub_scores = scores.topk(20)[1]
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            targets = target.numpy()
            for score, target in zip(sub_scores, targets):
                hit20.append(np.isin(target, score))
                if len(np.where(score == target)[0]) == 0:
                    mrr20.append(0)
                else:
                    mrr20.append(1 / (np.where(score == target)[0][0] + 1))

            sub_scores = scores.topk(10)[1]
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            for score, target in zip(sub_scores, targets):
                hit10.append(np.isin(target, score))
                if len(np.where(score == target)[0]) == 0:
                    mrr10.append(0)
                else:
                    mrr10.append(1 / (np.where(score == target)[0][0] + 1))
            
            sub_scores = scores.topk(5)[1]
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            for score, target in zip(sub_scores, targets):
                hit5.append(np.isin(target, score))
                if len(np.where(score == target)[0]) == 0:
                    mrr5.append(0)
                else:
                    mrr5.append(1 / (np.where(score == target)[0][0] + 1))

        result.append(np.mean(hit5) * 100)
        result.append(np.mean(hit10) * 100)
        result.append(np.mean(hit20) * 100)
        result.append(np.mean(mrr5) * 100)
        result.append(np.mean(mrr10) * 100)
        result.append(np.mean(mrr20) * 100)       


        hit20s, mrr20s = defaultdict(list), defaultdict(list)
        hit10s, mrr10s = defaultdict(list), defaultdict(list)
        hit5s, mrr5s = defaultdict(list), defaultdict(list)
        for data in test_loader:
            g, target = data
            g = g.to(torch.device('cuda'))
            targets = trans_to_cuda(target).long()
            scores =  model1(g, epoch) + model2(g, epoch)

            # scores = scores[-1] ###mix score
            assert not torch.isnan(scores).any()

            sub_scores = scores.topk(20)[1]
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            targets = target.numpy()
            for score, target in zip(sub_scores, targets):
                hit20s[target].append(np.isin(target, score))
                if len(np.where(score == target)[0]) == 0:
                    mrr20s[target].append(0)
                else:
                    mrr20s[target].append(1 / (np.where(score == target)[0][0] + 1))

            sub_scores = scores.topk(10)[1]
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            for score, target in zip(sub_scores, targets):
                hit10s[target].append(np.isin(target, score))
                if len(np.where(score == target)[0]) == 0:
                    mrr10s[target].append(0)
                else:
                    mrr10s[target].append(1 / (np.where(score == target)[0][0] + 1))
            
            sub_scores = scores.topk(5)[1]
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            for score, target in zip(sub_scores, targets):
                hit5s[target].append(np.isin(target, score))
                if len(np.where(score == target)[0]) == 0:
                    mrr5s[target].append(0)
                else:
                    mrr5s[target].append(1 / (np.where(score == target)[0][0] + 1))

        result.append(np.mean([np.mean(v) for v in hit5s.values()]) * 100)
        result.append(np.mean([np.mean(v) for v in hit10s.values()]) * 100)
        result.append(np.mean([np.mean(v) for v in hit20s.values()]) * 100)
        result.append(np.mean([np.mean(v) for v in mrr5s.values()]) * 100)
        result.append(np.mean([np.mean(v) for v in mrr10s.values()]) * 100)
        result.append(np.mean([np.mean(v) for v in mrr20s.values()]) * 100)
        return result

import pandas as pd
def read_sessions(filepath):
    sessions = pd.read_csv(filepath, sep='\t', header=None, squeeze=True)
    sessions = sessions.apply(lambda x: list(map(int, x.split(',')))).values
    return sessions

def read_dataset(dataset_dir):
    train_sessions = read_sessions(dataset_dir / 'train.txt')
    test_sessions = read_sessions(dataset_dir / 'test.txt')
    with open(dataset_dir / 'num_items.txt', 'r') as f:
        num_items = int(f.readline())
    return train_sessions, test_sessions, num_items

from pathlib import Path
from utils import AugmentedDataset,Mix_AugmentedDataset

if config.dataset in ['diginetica','gowalla','lastfm']:
    train_sessions, test_sessions, num_items = read_dataset(Path("/home/xl/lxl/model/SessionRec-pytorch/src/datasets/"+config.dataset))
    config.num_node = num_items
    if config.mixup:
        train_data = Mix_AugmentedDataset(train_sessions, training=True, train_len=len(train_sessions), unique=config.unique, add_self_loop=config.add_self_loop)
    else:
        train_data = AugmentedDataset(train_sessions, training=True, train_len=len(train_sessions), unique=config.unique, add_self_loop=config.add_self_loop)
    test_data = AugmentedDataset(test_sessions, training=False, train_len=len(train_sessions), unique=config.unique, add_self_loop=config.add_self_loop)
else:
    train_sessions = pickle.load(open('/home/xl/lxl/dataset/' + config.dataset + "/" +config.dataset + '/all_train_seq.txt', 'rb'))
    if config.mixup:
        train_data = Mix_AugmentedDataset(train_sessions, training=True, train_len=len(train_sessions), unique=config.unique, add_self_loop=config.add_self_loop)
    else:
        train_data = AugmentedDataset(train_sessions, training=True, train_len=len(train_sessions), unique=config.unique, add_self_loop=config.add_self_loop)
    test_data = pickle.load(open('/home/xl/lxl/dataset/' + config.dataset + "/" +config.dataset + '/test.txt', 'rb'))
    test_data = Data(test_data, is_train=False, unique=config.unique, add_self_loop=config.add_self_loop)


if __name__ == "__main__":
    print(config.dataset, config.num_node, "lr:",config.lr, "lr_dc:",config.lr_dc, "lr_dc_step:",config.lr_dc_step, "dropout_local:",config.dropout_local, "feat_drop:",config.feat_drop, "label_smooth:",config.lb_smooth, "window_size:", config.window_size, "unique:", config.unique, "add_self_loop:", config.add_self_loop)
    init_seed(42)

    # train_data = pickle.load(open('/home/xl/lxl/dataset/' + config.dataset + "/" +config.dataset + '/train.txt', 'rb'))
    # test_data = pickle.load(open('/home/xl/lxl/dataset/' + config.dataset + "/" +config.dataset + '/test.txt', 'rb'))
    # edge2idx = pickle.load(open('/home/xl/lxl/dataset/' + config.dataset + "/" +config.dataset + '/edge2idx.pkl', 'rb'))
    # edge2fre = pickle.load(open('/home/xl/lxl/dataset/' + config.dataset + "/" +config.dataset + '/edge2fre.pkl', 'rb'))
    # adj = pickle.load(open('/home/xl/lxl/model/DGL/data/'+config.dataset+'_adj.pkl', 'rb'))
    # train_data = Data(train_data, edge2idx, edge2fre, adj, is_train=True)
    # test_data = Data(test_data, edge2idx, edge2fre, adj, is_train=False)

    model1 = trans_to_cuda(Ensamble2(num_node = config.num_node))
    model2 = trans_to_cuda(Ensamble2(num_node = config.num_node))
    # checkpoint = torch.load('/home/xl/lxl/model/DGL/data/'+config.dataset+"_model.pkl")
    # model_dict = model.state_dict()
    # state_dict = {k:v for k,v in checkpoint.items() if k in model_dict.keys()}
    # model.load_state_dict(state_dict)
    
    start = time.time()
    best_result = [0, 0, 0, 0, 0, 0]
    best_epoch = [0, 0, 0, 0, 0, 0]
    bad_counter = 0

    for epoch in range(config.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit5, hit10, hit20, mrr5, mrr10, mrr20, hit5s, hit10s, hit20s, mrr5s, mrr10s, mrr20s = train_test(model1, model2, train_data, test_data, epoch, train_sessions)
        if hit5 >= best_result[0]:
            best_result[0] = hit5
            best_epoch[0] = epoch
        if hit10 >= best_result[1]:
            best_result[1] = hit10
            best_epoch[1] = epoch
        if hit20 >= best_result[2]:
            best_result[2] = hit20
            best_epoch[2] = epoch
        if mrr5 >= best_result[3]:
            best_result[3] = mrr5
            best_epoch[3] = epoch
        if mrr10 >= best_result[4]:
            best_result[4] = mrr10
            best_epoch[4] = epoch
        if mrr20 >= best_result[5]:
            best_result[5] = mrr20
            best_epoch[5] = epoch
        print('Current Result:')
        print('\tRecall5s:\t%.4f\tRecall10s:\t%.4f\tRecall@20s:\t%.4f\tMMR@5s:\t%.4f\tMMR@10s:\t%.4f\tMMR@20s:\t%.4f' % (hit5s, hit10s, hit20s, mrr5s, mrr10s, mrr20s))
        print('\tRecall5:\t%.4f\tRecall10:\t%.4f\tRecall@20:\t%.4f\tMMR@5:\t%.4f\tMMR@10:\t%.4f\tMMR@20:\t%.4f' % (hit5, hit10, hit20, mrr5, mrr10, mrr20))
        print('Best Result:')
        print('\tRecall5:\t%.4f\tRecall10:\t%.4f\tRecall@20:\t%.4f\tMMR@5:\t%.4f\tMMR@10:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d,\t%d,\t%d,\t%d,\t%d' % (
            best_result[0], best_result[1], best_result[2], best_result[3],  best_result[4], best_result[5], best_epoch[0], best_epoch[1], best_epoch[2], best_epoch[3], best_epoch[4], best_epoch[5]))

        
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))