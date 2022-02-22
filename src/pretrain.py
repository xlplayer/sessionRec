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
from model import Pretrain_SessionGraph
from utils import Data
import torch.nn.functional as F
import networkx as nx
import scipy.sparse as sp
from sklearn.cluster import KMeans
import dgl


def init_seed(seed=None):
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


def collate_fn(samples):
    g1, g2 = zip(*samples)
    g1 = dgl.batch(g1)
    g2 = dgl.batch(g2)
    return g1, g2

def pretrain(model, train_data, epoch, train_sessions):
    print('start training: ', datetime.datetime.now())
    model.train()

    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=20, batch_size=config.batch_size,
                                               shuffle=False, pin_memory=True, collate_fn=collate_fn)
    
    with tqdm(train_loader) as t:
        for data in t:
            model.optimizer.zero_grad()
            g1, g2 = data
            g1 = g1.to(torch.device('cuda'))
            g2 = g2.to(torch.device('cuda'))
            loss =  model(g1, g2)
            t.set_postfix(
                        loss = loss.item(),
                        lr = model.optimizer.state_dict()['param_groups'][0]['lr'])
            loss.backward()
            model.optimizer.step()
            total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)

    model.scheduler.step()

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
from utils import Pretrain_AugmentedDataset
train_sessions, test_sessions, num_items = read_dataset(Path("/home/xl/lxl/model/SessionRec-pytorch/src/datasets/diginetica"))
config.num_node = num_items
G = pickle.load(open('/home/xl/lxl/model/DGL/data/'+config.dataset+'_adj.pkl', 'rb'))
train_data = Pretrain_AugmentedDataset(train_sessions)

if __name__ == "__main__":
    print(config.dataset, config.num_node, "lr:",config.lr, "lr_dc:",config.lr_dc, "lr_dc_step:",config.lr_dc_step, "dropout_local:",config.dropout_local)
    init_seed(42)

    # train_data = pickle.load(open('/home/xl/lxl/dataset/' + config.dataset + "/" +config.dataset + '/train.txt', 'rb'))
    # test_data = pickle.load(open('/home/xl/lxl/dataset/' + config.dataset + "/" +config.dataset + '/test.txt', 'rb'))
    # edge2idx = pickle.load(open('/home/xl/lxl/dataset/' + config.dataset + "/" +config.dataset + '/edge2idx.pkl', 'rb'))
    # edge2fre = pickle.load(open('/home/xl/lxl/dataset/' + config.dataset + "/" +config.dataset + '/edge2fre.pkl', 'rb'))
    # adj = pickle.load(open('/home/xl/lxl/model/DGL/data/'+config.dataset+'_adj.pkl', 'rb'))
    # train_data = Data(train_data, edge2idx, edge2fre, adj, is_train=True)
    # test_data = Data(test_data, edge2idx, edge2fre, adj, is_train=False)

    model = trans_to_cuda(Pretrain_SessionGraph(num_node = config.num_node))

    
    start = time.time()
    best_result = [0, 0, 0, 0, 0, 0]
    best_epoch = [0, 0, 0, 0, 0, 0]
    bad_counter = 0

    for epoch in range(config.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        pretrain(model, train_data, epoch, train_sessions)
        torch.save(model.state_dict(),  '/home/xl/lxl/model/DGL/data/'+config.dataset+"_model.pkl")

        
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))