import dgl
import numpy as np


from utils import *
from idle_gpu import idle_gpu
from dgl.nn import RelGraphConv
from model import Model, DotPredictor


if __name__ == "__main__":
    test_path = "./data/test"
    GTEx_file = "/GTEx.txt"
    links_file = "/links.txt"
    GO_file = "/GO.txt"
    test_file = "/test_data.txt"
    model_file = "./save/embedding.out"
    pred_file = "./save/pred.out"
    if torch.cuda.is_available():
        dev = torch.device(f'cuda:{idle_gpu()}')
    else:  
        dev = torch.device('cpu')

    id2name, name2id = idMap(test_path + GO_file)
    g = constructGraph(test_path + links_file, name2id)[0]
    g.ndata['feature'] = torch.cat(
        [loadGTEx(test_path + GTEx_file, name2id), loadGO(test_path + GO_file, name2id)],
        dim=1
    )

    test_pos_g = load_test(test_path + test_file, name2id, g.number_of_nodes())

    g = g.to(dev)
    model = Model(g.ndata['feature'].shape[1], 64, 2, 2)
    # model = RelGraphConv(g.ndata['feature'].shape[1], 8, 2, low_mem=True).to(dev)
    pred = DotPredictor()
    model = torch.load(model_file).to(dev)
    pred = torch.load(pred_file).to(dev)

    testLoader = dgl.dataloading.EdgeDataLoader(
        test_pos_g,
        test_pos_g.edges('eid'),
        block_sampler=dgl.dataloading.MultiLayerFullNeighborSampler(2),
        negative_sampler=dgl.dataloading.negative_sampler.Uniform(1),
        batch_size=test_pos_g.number_of_edges(),
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        device=dev
    )

    for input_nodes, pos_g, neg_g, blocks in testLoader:
        with torch.no_grad():
            h = model(pos_g, g.ndata['feature'][pos_g.ndata['_ID']], pos_g.edata['rel_type'])
            pos_score = pred(pos_g, h)
            neg_score = pred(neg_g, h)
            print('AUC', compute_auc(pos_score, neg_score))
        break



