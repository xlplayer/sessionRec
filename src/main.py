import config
import time
import datetime
import torch
import numpy as np
import random
import pickle
from tqdm import tqdm
from model import SessionGraph
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
    g, target = zip(*samples)
    g = dgl.batch(g)
    return g, torch.tensor(target)

def train_test(model, train_data, test_data, epoch):
    print('start training: ', datetime.datetime.now())
    model.train()

    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=20, batch_size=config.batch_size,
                                               shuffle=True, pin_memory=True, collate_fn=collate_fn)
    
    with tqdm(train_loader) as t:
        for data in t:
            model.optimizer.zero_grad()
            g, target = data
            g = g.to(torch.device('cuda'))
            targets = trans_to_cuda(target).long()
            scores =  model(g)

            loss = model.loss_function(scores, targets - 1)
            t.set_postfix(loss = loss.item(), lr = model.optimizer.state_dict()['param_groups'][0]['lr'])
            loss.backward()
            model.optimizer.step()
            total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)

    model.scheduler.step()

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=20, batch_size=config.batch_size,
                                              shuffle=False, pin_memory=True, collate_fn=collate_fn)
    result = []
    hit20, mrr20 = [], []
    hit10, mrr10 = [], []
    hit5, mrr5 = [], []
    for data in test_loader:
        g, target = data
        g = g.to(torch.device('cuda'))
        targets = trans_to_cuda(target).long()
        scores =  model(g)

        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        targets = target.numpy()
        for score, target in zip(sub_scores, targets):
            hit20.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr20.append(0)
            else:
                mrr20.append(1 / (np.where(score == target - 1)[0][0] + 1))

        sub_scores = scores.topk(10)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target in zip(sub_scores, targets):
            hit10.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr10.append(0)
            else:
                mrr10.append(1 / (np.where(score == target - 1)[0][0] + 1))
        
        sub_scores = scores.topk(5)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target in zip(sub_scores, targets):
            hit5.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr5.append(0)
            else:
                mrr5.append(1 / (np.where(score == target - 1)[0][0] + 1))

    result.append(np.mean(hit5) * 100)
    result.append(np.mean(hit10) * 100)
    result.append(np.mean(hit20) * 100)
    result.append(np.mean(mrr5) * 100)
    result.append(np.mean(mrr10) * 100)
    result.append(np.mean(mrr20) * 100)       

    return result

if __name__ == "__main__":
    print(config.dataset, config.num_node, "lr:",config.lr, "lr_dc:",config.lr_dc, "lr_dc_step:",config.lr_dc_step, "dropout_local:",config.dropout_local)
    init_seed(42)

    train_data = pickle.load(open('/home/xl/lxl/dataset/' + config.dataset + "/" +config.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('/home/xl/lxl/dataset/' + config.dataset + "/" +config.dataset + '/test.txt', 'rb'))
    edge2idx = pickle.load(open('/home/xl/lxl/dataset/' + config.dataset + "/" +config.dataset + '/edge2idx.pkl', 'rb'))
    train_data = Data(train_data, edge2idx)
    test_data = Data(test_data, edge2idx)

    model = trans_to_cuda(SessionGraph())

    start = time.time()
    best_result = [0, 0, 0, 0, 0, 0]
    best_epoch = [0, 0, 0, 0, 0, 0]
    bad_counter = 0

    for epoch in range(config.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit5, hit10, hit20, mrr5, mrr10, mrr20 = train_test(model, train_data, test_data, epoch)
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
        print('\tRecall5:\t%.4f\tRecall10:\t%.4f\tRecall@20:\t%.4f\tMMR@5:\t%.4f\tMMR@10:\t%.4f\tMMR@20:\t%.4f' % (hit5, hit10, hit20, mrr5, mrr10, mrr20))
        print('Best Result:')
        print('\tRecall5:\t%.4f\tRecall10:\t%.4f\tRecall@20:\t%.4f\tMMR@5:\t%.4f\tMMR@10:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d,\t%d,\t%d,\t%d,\t%d' % (
            best_result[0], best_result[1], best_result[2], best_result[3],  best_result[4], best_result[5], best_epoch[0], best_epoch[1], best_epoch[2], best_epoch[3], best_epoch[4], best_epoch[5]))

        
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))