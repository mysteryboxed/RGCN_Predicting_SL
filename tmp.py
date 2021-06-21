import dgl
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import json
import random
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

# 根据PPI数据集创建图，添加边的feature（环和普通边）
def constructGraph(link_file, name2id, use_map=False):
    print(f"constructing graph")
    #if os.path.exists("./graph.gph"):
    #    return load_graphs("./graph.gph")[0]

    links = np.loadtxt(link_file, dtype='str', delimiter='\t')

    if name2id is True:
        u_nodes = [name2id[u] for u in links[:, 0]]
        v_nodes = [name2id[v] for v in links[:, 1]]
    else:
        u_nodes = links[:, 0].astype(int)
        v_nodes = links[:, 1].astype(int)

    num_nodes = len(name2id)
    g = dgl.graph((u_nodes, v_nodes), num_nodes=num_nodes)


    rel_type = torch.LongTensor(links[:, 2].astype(int)) # converting type is a must
    g.edata["rel_type"] = rel_type
    print(rel_type)

    edge_tri = g.edges(form='all')
    print(f"u\n{u_nodes[:5]}\nv\n{v_nodes[:5]}")
    print(f"edata first 10\n{g.edata['rel_type'][:10]}")
    print(f"edata last 10\n{g.edata['rel_type'][-10:]}")
    print(f"edge info\n{edge_tri[:10]}")

    print(f"graph\n{g}")
    save_graphs("graph.gph", [g])
    return [g]


raw_path = "./data/raw"
train_path = "./data/train"
test_path = "./data/test"

bp_file = "/bp.txt"
cc_file = "/cc.txt"
mf_file = "/mf.txt"
gr_file = "/gra.txt"
sl_file = "/sl.txt"
sln_file = "/sln.txt"
na_file = "/name2id.txt"

with open(train_path + na_file, 'r') as read_file:
    name2id = json.load(read_file)

print(name2id[0,:])
