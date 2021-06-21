import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn import RelGraphConv

class MLPPrediction(nn.Module): 
    def __init__(self, in_feat):
        super(MLPPrediction, self).__init__()
        self.liner1 = nn.Linear(in_feat * 2, in_feat)
        self.liner2 = nn.Linear(in_feat, 1)

    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        # print(f"feature cat {h.shape}")
        return {'score': self.liner2(F.relu(self.liner1(h)))}

    def forward(self, g, feature):
        with g.local_scope():
            g.ndata['h'] = feature
            g.apply_edges(self.apply_edges)
            return g.edata['score']

class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            # return g.edata['score']
            return torch.sigmoid(g.edata['score'])

class Model(nn.Module):
    def __init__(self, in_feat, h_feat, out_feat, num_rel, num_layer):
        super(Model, self).__init__()
        self.in_feat = in_feat
        self.h_feat = h_feat
        self.out_feat = out_feat
        self.num_rel = num_rel
        self.num_layer = num_layer

        self.layers = nn.ModuleList()
        self.layers.append(RelGraphConv(self.in_feat, self.h_feat, self.num_rel, low_mem=True))
        for _ in range(self.num_layer):
            self.layers.append(RelGraphConv(self.h_feat, self.h_feat, self.num_rel, low_mem=True))
        self.layers.append(RelGraphConv(self.h_feat, self.out_feat, self.num_rel, low_mem=True))

    def forward(self, g, feat, edges):
        for layer in self.layers:
            feat = layer(g, feat, edges)
            feat = F.relu(feat)
        return feat


class RGCN(nn.Module):
    def __init__(self, in_feat, out_feat, num_rel):
        super(RGCN, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rel = num_rel

        self.weight = nn.Parameter(torch.Tensor(num_rel, in_feat, out_feat))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))


    def forward(self, g, h):
        def message_func(edges):
            w = self.weight[edges.data['rel_type']]
            msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
            # print(f"msg{msg.shape}, edge {edges.data['norm'].shape} ")
            # msg = msg * edges.data['norm']
            return {'msg': msg}


        def apply_func(nodes):
            pass

        with g.local_scope():
            g.ndata['h'] = h
            g.update_all(message_func=message_func, reduce_func=fn.sum(msg='msg', out='h'))
            return g.ndata['h']
