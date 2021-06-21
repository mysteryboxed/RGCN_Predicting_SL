import numpy as np
from tqdm import tqdm
import json
import random
from shutil import copyfile
import torch
from pyrwr.ppr import PPR
from utils_ import *
from multiprocessing import Pool
from functools import partial
from time import *


#return the dij calculated distance
def dij_distance(graph, u, max_dis):
    torch.set_printoptions(threshold=10_000)
    visited = [0] * graph.number_of_nodes()
    distance = [max_dis] * graph.number_of_nodes()
    distance[u] = 0
    add = u
    edges = torch.cat([graph.edges()[0].unsqueeze(1), graph.edges()[1].unsqueeze(1)], dim=1)
    rel_type = torch.unsqueeze(graph.edata["rel_type"], dim=1)
    #print(torch.cat([edges, rel_type], dim=1))
    #print(graph.ndata['_ID'])

    for i in range(graph.number_of_nodes()): # if all visited, than stoped
        visited[add] = 1
        # update the distance
        neighbors = graph.out_edges(add)[1]
        for neighbor in neighbors:
            if distance[neighbor] > distance[add] + 1:
                distance[neighbor] = distance[add] + 1

        # updata the node
        min_dis = max_dis
        for node in range(graph.number_of_nodes()):
            if distance[node] <= min_dis and not visited[node]:
                add = node
                min_dis = distance[node]
        #print(distance)

    return distance


# generate the subgraph for node_list
# c: restart probability
# ratio: select the node above the ratio * avg
def gen_subgraph(node_list, ppr, graph, pn_path, ratio=5, c = 0.15, one_hot_length=40):
    ans = ppr.compute(node_list, c)
    idx = np.where(ans > ratio * np.mean(ans))[0]
    sub = dgl.node_subgraph(graph, idx, store_ids=True)

    #print(sub)
    #print(node_list)

    # get the u, v subgraph id
    u = int((sub.ndata["_ID"] == node_list[0]).nonzero(as_tuple=True)[0])
    v = int((sub.ndata["_ID"] == node_list[1]).nonzero(as_tuple=True)[0]) #此处应该是node_list[1]
    #print(f"sub node id {u} {v}")

    # delete the edge u, v
    sub_rm_edge = dgl.remove_edges(sub, torch.tensor([u, v]))
    dis_u = dij_distance(sub_rm_edge, u, int(one_hot_length / 2) - 1)
    dis_v = dij_distance(sub_rm_edge, v, int(one_hot_length / 2) - 1)

    sub.ndata['fet_u'] = torch.Tensor(dis_u)
    sub.ndata['fet_v'] = torch.Tensor(dis_v)

    save_graphs('./save_sub/'+pn_path+'/'+str(node_list[0])+'_'+str(node_list[1])+'.bin', sub)


if __name__ == "__main__":
    raw_path = "./data/raw"
    train_path = "./data/train"
    test_path = "./data/test"

    bp_file = "/bp.txt"
    cc_file = "/cc.txt"
    mf_file = "/mf.txt"
    gr_file = "/gra.txt"
    sl_file = "/sl.txt"
    na_file = "/name2id.txt"

    with open(train_path + na_file, 'r') as read_file:
        name2id = json.load(read_file)

    ppi_graph = constructGraph(train_path + gr_file, name2id)[0]
    print(ppi_graph)

    ppr = PPR()
    ppr.read_graph(train_path + gr_file, "directed")
    links_p = np.loadtxt(train_path + sl_file, dtype=int)
    links_n = gen_sln(train_path + sl_file, train_path)
    print(f"pos link {len(links_p)}")
    print(f"neg link {len(links_n)}")

    partial_gen1 = partial(gen_subgraph, ppr=ppr, graph=ppi_graph,pn_path='pos')
    pool1 = Pool(processes=40)
    pool1.map(partial_gen1, links_p)

    partial_gen2 = partial(gen_subgraph, ppr=ppr, graph=ppi_graph,pn_path='neg')
    pool2 = Pool(processes=40)
    pool2.map(partial_gen2, links_n)

    print(f"pos links len {len(links_p)}")
    print(f"neg links len {len(links_n)}")

    # begin_time=time()
    # i=0
    # for link in links_p:
    #   gen_subgraph(links_p[i], ppr, ppi_graph, pn_path='pos')
    #   i+=1
    #   if(i==100):
    #     break
    # end_time=time()
    # print("time=",end_time-begin_time)