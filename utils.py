import numpy as np
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random
import itertools
import os
from dgl.data.utils import load_graphs, save_graphs
from sklearn.metrics import roc_auc_score, average_precision_score

def idMap(feature_file):
    name = np.loadtxt(feature_file, dtype=str, delimiter='\t')[:, 0]
    name = np.squeeze(name)
    id2name = {}
    name2id = {}
    i = 0
    for n in name:
        id2name[i] = n
        name2id[n] = i
        i += 1
    return id2name, name2id

# 根据PPI创建图，添加边的feature（环和普通边）
def constructGraph(link_file, name2id):
    print(f"constructing graph")
    if os.path.exists("./graph.gph"):
        return load_graphs("./graph.gph")[0]

    links = np.loadtxt(link_file, dtype='str', delimiter='\t')

    u_nodes = [name2id[u] for u in links[:, 0]]
    v_nodes = [name2id[v] for v in links[:, 1]]

    num_nodes = len(name2id)
    g = dgl.graph((u_nodes, v_nodes), num_nodes=num_nodes)


    rel_type = torch.LongTensor(links[:, 2].astype(int))
    g.edata["rel_type"] = rel_type

    edge_tri = g.edges(form='all')
    print(f"u\n{u_nodes[:5]}\nv\n{v_nodes[:5]}")
    print(f"edata first 10\n{g.edata['rel_type'][:10]}")
    print(f"edata last 10\n{g.edata['rel_type'][-10:]}")
    print(f"edge info\n{edge_tri[:10]}")

    print(f"graph\n{g}")
    save_graphs("graph.gph", [g])
    return [g]

# 返回节点的GO特征矩阵
def load_feature(feature_file, name2id):
    feature = np.loadtxt(feature_file, dtype=str, delimiter='\t') #以tab为分隔符，分别载入每一行的特征embedding数值
    rows = len(name2id) #name2id的长度指的就是有多少个基因
    cols = feature.shape[1] - 1 #每个基因有多少个特征(-1是因为每一行中第一列是基因名字)

    result = np.zeros((rows, cols))
    for line in feature:
        result[name2id[line[0]]] = line[1:]  #将所有基因的名字转为其对应的编号，再载入特征到每个结果中
    print(f"loading feature {result.shape}")#返回的result为：编号对应的所有embedding数值。
    return torch.Tensor(result)#生成单精度浮点类型的张量

# 返回节点的GTEx特征矩阵
def loadGTEx(feature_file, name2id):
    
    feature = np.loadtxt(feature_file, dtype=str, delimiter='\t')
    rows = len(name2id)
    cols = feature.shape[1] - 1

    result = np.zeros((rows, cols))
    for line in feature:
        result[name2id[line[0]]] = line[1:] 
    print(f"loading GTEx feature {result.shape}")
    return torch.Tensor(result)

def split_data(train_file, name2id, num_nodes, train_ratio, valid_ratio):
    train_data = np.loadtxt(train_file, dtype=str, delimiter='\t')
    for i in range(len(train_data)):#首先，把所有的基因名字都改为对应的基因编码-name2id
        for j in range(len(train_data[0])):
            train_data[i][j] = int(name2id[train_data[i][j]])

    train_data = train_data.astype(int)
    # np.random.shuffle(train_data)

    num_train = int(train_data.shape[0] * train_ratio)
    num_valid = train_data.shape[0] - num_train
    print(f"num_train {num_train} num_valid {num_valid}")

    u_nodes = np.squeeze(train_data[:, 0])
    v_nodes = np.squeeze(train_data[:, 1])

    train_pos_g = dgl.graph((u_nodes[:num_train], v_nodes[:num_train]), num_nodes=num_nodes)
    valid_pos_g = dgl.graph((u_nodes[num_train:], v_nodes[num_train:]), num_nodes=num_nodes)
    train_pos_g.add_edges(range(num_nodes), range(num_nodes))
    valid_pos_g.add_edges(range(num_nodes), range(num_nodes))

    train_pos_g.edata['rel_type'] = torch.cat([torch.ones(num_train), torch.zeros(num_nodes)])
    valid_pos_g.edata['rel_type'] = torch.cat([torch.ones(num_valid), torch.zeros(num_nodes)])

    edge_tri = train_pos_g.edges(form='all')
    # print(f"edata first 10\n{train_pos_g.edata['rel_type'][:10]}")
    # print(f"edata last 10\n{train_pos_g.edata['rel_type'][-10:]}")
    # print(f"edge info\n{edge_tri[:10]}")
    # exit()
 
    return train_pos_g, valid_pos_g

def load_test(test_file, name2id, num_nodes):
    test_data = np.loadtxt(test_file, dtype=str, delimiter='\t')
    for i in range(len(test_data)):
        for j in range(len(test_data[0])):
            test_data[i][j] = int(name2id[test_data[i][j]])

    test_data = test_data.astype(int)

    num_test = test_data.shape[0]
    print(f"num_test {num_test}")

    u_nodes = np.squeeze(test_data[:, 0])
    v_nodes = np.squeeze(test_data[:, 1])

    test_pos_g = dgl.graph((u_nodes, v_nodes), num_nodes=num_nodes)
    test_pos_g.add_edges(range(num_nodes), range(num_nodes))

    test_pos_g.edata['rel_type'] = torch.cat([torch.zeros(num_test), torch.ones(num_nodes)])

    return test_pos_g


def compute_loss(pos_score, neg_score, loss_func, dev):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape), torch.zeros(neg_score.shape)]).to(dev)
    # print(f"scores {scores}")
    # print(f"labels {labels}")
    return loss_func(scores, labels)

def margin(pos_score, neg_score, dev):
    # Margin loss
    n_edges = pos_score.shape[0]
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()

def bpr(pos_score, neg_score):
    # print(f"loss\n{torch.sum(-torch.log(F.sigmoid(pos_score - neg_score)))}")
    # print(f"ten\n{(pos_score - neg_score)[:10]}")
    # exit()
    return torch.sum(-torch.log(F.sigmoid(pos_score - neg_score)))

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).cpu().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)

def compute_auprc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).cpu().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).numpy()
    return average_precision_score(labels, scores)

def real_time_auc(pos_score, neg_score):
    pos = np.array(pos_score.clone().detach().cpu())
    neg = np.array(neg_score.clone().detach().cpu())

    scores = np.concatenate([pos, neg])
    labels = np.concatenate(
        [np.ones(pos.shape[0]), np.zeros(neg.shape[0])]
    )
    return roc_auc_score(labels, scores)

    


