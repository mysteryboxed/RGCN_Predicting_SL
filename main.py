import dgl
import torch
import torch.nn as nn
from dgl.nn import RelGraphConv

from utils import *
from model import Model, MLPPrediction, DotPredictor
from idle_gpu import idle_gpu

if __name__ == "__main__":
    train_path = "./data/train"
    bp_file = "/bp.txt"
    cc_file = "/cc.txt"
    mf_file = "/mf.txt"
    gr_file = "/gra.txt"
    sl_file = "/sl.txt"

    model_file = "./save/embedding.out"
    pred_file = "./save/pred.out"

    if torch.cuda.is_available():
        dev = torch.device(f'cuda:{idle_gpu()}')
    else:  
        dev = torch.device('cpu')

    id2name, name2id = idMap(train_path + bp_file)
    g = constructGraph(train_path + gr_file, name2id)[0]

    g.ndata['feature'] = torch.cat(
        [
            load_feature(train_path + bp_file, name2id),
            load_feature(train_path + cc_file, name2id),
            load_feature(train_path + mf_file, name2id)
        ],
        dim=1
    )

    train_pos_g, valid_pos_g = \
        split_data(train_path + sl_file, name2id, g.number_of_nodes(), train_ratio=0.9, valid_ratio=0.1)

    g = g.to(dev)
    print(g)
    model = Model(g.ndata['feature'].shape[1], 128, 128, 7, 1).to(dev)
    # model = RelGraphConv(g.ndata['feature'].shape[1], 8, 2, low_mem=True).to(dev)
    pred = DotPredictor().to(dev)  #pred是一个sigmoid函数用于预测结果
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.001)

    #ppi是构造出来的数据集，sl是需要预测的

    trainLoader = dgl.dataloading.EdgeDataLoader(
        train_pos_g,
        train_pos_g.edges('eid'),
        block_sampler=dgl.dataloading.MultiLayerFullNeighborSampler(2),
        negative_sampler=dgl.dataloading.negative_sampler.Uniform(1),
        batch_size=4096,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        device=dev
    )
    validLoader = dgl.dataloading.EdgeDataLoader(
        valid_pos_g,
        valid_pos_g.edges('eid'),
        block_sampler=dgl.dataloading.MultiLayerFullNeighborSampler(2),
        negative_sampler=dgl.dataloading.negative_sampler.Uniform(1),
        batch_size=valid_pos_g.number_of_edges(),
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        device=dev
    )
    score = 0
    for epoch in range(300):
        for input_nodes, pos_g, neg_g, blocks in trainLoader:
            h = model(pos_g, g.ndata['feature'][pos_g.ndata['_ID']], pos_g.edata['rel_type'])
            # print(f"input nodes{input_nodes}")
            # print(f"pos g\n{pos_g}")
            # print(f"neg g\n{neg_g}")
            # print(f"block\n{blocks}")
            # print(f"pos g id\n{pos_g.ndata['_ID']}")
            # print(f"pos nodes\n{pos_g.nodes()}")
            # print(f"h\n{h[input_nodes.to('cpu')]}")
            # print(f"pos edge\n{pos_g.edges()}")
            # print(f"neg edge\n{neg_g.edges()}")
            pos_score = pred(pos_g, h)
            neg_score = pred(neg_g, h)

            # print(f"embedding h\n{h}")
            # print(f"pos {pos_score[:5]}")
            # print(f"neg {neg_score[:5]}")
            loss = compute_loss(pos_score, neg_score, F.binary_cross_entropy_with_logits, dev)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(f"in epoch {epoch}, loss {loss}")
            # print(f"in epoch {epoch}, loss {loss} auc {real_time_auc(pos_score, neg_score)}")

            for input_nodes, pos_g, neg_g, blocks in validLoader:
                with torch.no_grad():
                    h = model(pos_g, g.ndata['feature'][pos_g.ndata['_ID']], pos_g.edata['rel_type'])
                    pos_score = pred(pos_g, h)
                    neg_score = pred(neg_g, h)
                    new_score = compute_auc(pos_score, neg_score)
                    new_auprc = compute_auprc(pos_score, neg_score)
                    print(f"epoch {epoch}, AUC {new_score}, AUPRC {new_auprc}")
                    if score < new_score:
                        score = new_score
                        print(f"epoch {epoch} save model, score {score}")
                        torch.save(model, model_file)
                        torch.save(pred, pred_file)
                break
    print(f"score {score}")
