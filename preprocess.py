import numpy as np
from tqdm import tqdm
import json
import random
from shutil import copyfile
import torch

# return a map of feature name->embedding
def load_bp(bp_file):
    return torch.load(bp_file)

def load_cc(cc_file):
    return torch.load(cc_file)

def load_mf(mf_file):
    return torch.load(mf_file)

# array of links [gene, gene]
def load_bi(bi_file):
    raw_bi = np.loadtxt(bi_file, dtype=str, delimiter='\t', skiprows=1)
    bi = []
    for line in raw_bi:
        if line[-1] == "physical": bi.append([line[0], line[1]])
    return np.array(bi)

# array of links [gene, type, gene]
def load_re(re_file):
    return np.loadtxt(re_file, dtype=str, delimiter='\t')

# map alias->name
def load_na(na_file):
    with open(na_file) as read_file:
        return json.load(read_file)

# array of links [gene, gene]
def load_sl(sl_file):
    useless_method = {
        'GenomeRNAi;Text Mining',
        'Text Mining',
        'computer analysis',
        'Synlethality;Text Mining',
        'textmining',
        'Text Mining;Synlethality',
        'Decipher;Text Mining',
        'computational analysis',
        'Text Mining;Daisy'}
    sl_list = []
    with open(sl_file, 'r') as readfile:
        for line in readfile.readlines():
            tmp = line.split('\t')
            if tmp[-2] not in useless_method:
                sl_list.append([tmp[0], tmp[2]])
    return np.array(sl_list)

def basic_info(bp, cc, mf, bio, reactome, std_name, sl):
    print("{:<5}{:<10}{:<10}".format("name", "nodes", "links"))
    print("{:<5}{:<10}".format("bp", len(bp)))
    print("{:<5}{:<10}".format("cc", len(cc)))
    print("{:<5}{:<10}".format("mf", len(mf)))
    print("{:<5}{:<10}{:<10}".format(
        "bio", len(set(bio[:,0]).union(set(bio[:,1]))), len(bio)
    ))
    print("{:<5}{:<10}{:<10}".format(
        "reac", len(set(reactome[:,0]).union(set(reactome[:,2]))), len(reactome)
    ))
    print("{:<5}{:<10}{:<10}".format(
        "SL", len(set(sl[:,0]).union(set(sl[:,1]))), len(sl)
    ))

def convert_name(bp, cc, mf, bio, reactome, std_name, sl):
    bp_result = {}
    for key in bp.keys():
        if key in std_name.keys():
            bp_result[std_name[key]] = bp[key]
        else: 
            bp_result[key] = bp[key]
    cc_result = {}
    for key in cc.keys():
        if key in std_name.keys():
            cc_result[std_name[key]] = cc[key]
        else:
            cc_result[key] = cc[key]
    mf_result = {}
    for key in mf.keys():
        if key in std_name.keys():
            mf_result[std_name[key]] = mf[key]
        else:
            mf_result[key] = mf[key]
    
    for line in bio:
        if line[0] in std_name.keys(): line[0] = std_name[line[0]]
        if line[1] in std_name.keys(): line[1] = std_name[line[1]]
    for line in reactome:
        if line[0] in std_name.keys(): line[0] = std_name[line[0]]
        if line[2] in std_name.keys(): line[2] = std_name[line[2]]
    for line in sl:
        if line[0] in std_name.keys(): line[0] = std_name[line[0]]
        if line[1] in std_name.keys(): line[1] = std_name[line[1]]
    return bp_result, cc_result, mf_result, bio, reactome, std_name, sl

# return map name -> array
def normalize_map(feature):
    values = []
    result = {}
    for key in feature.keys():
        values.append([key] + feature[key].tolist())
    values = np.array(values)
    values_data = normalize_arr(values[:, 1:].astype(float))
    i = 0
    for key in values[:, 0]:
        result[key] = values_data[i]
        i += 1
    return result

# get array of float return array of float
def normalize_arr(unnorm):
    normed = unnorm
    normed = (normed - normed.mean(axis=0)) / normed.std(axis=0)
    return np.nan_to_num(normed)

# return list [name, feature ...]
def append_missing(bp, cc, mf, sl):
    name_set = set(sl[:, 0]).union(set(sl[:, 1]))
    bp_list = []
    cc_list = []
    mf_list = []

    values = []
    for key in bp.keys():
        values.append(bp[key].tolist())
    bp_mean = np.array(values).mean(axis=0)
    values = []
    for key in cc.keys():
        values.append(cc[key].tolist())
    cc_mean = np.array(values).mean(axis=0)
    values = []
    for key in mf.keys():
        values.append(mf[key].tolist())
    mf_mean = np.array(values).mean(axis=0)

    for name in list(name_set):
        if name not in bp.keys():
            bp_list.append([name] + bp_mean.tolist())
        else:
            bp_list.append([name] + bp[name].tolist())
        if name not in cc.keys():
            cc_list.append([name] + cc_mean.tolist())
        else:
            cc_list.append([name] + cc[name].tolist())
        if name not in mf.keys():
            mf_list.append([name] + mf_mean.tolist())
        else:
            mf_list.append([name] + mf[name].tolist())

    return bp_list, cc_list, mf_list

'''
0: self
1: activate
2: be activated
3: mutual
4: inhibit
5: be inhibitted
6: bio
return a list of [name, name, rel_type]
'''
def merge_graph(bio, reactome, sl):
    result_graph = []
    name_set = set(sl[:, 0]).union(set(sl[:, 1]))
    for name in name_set:
        result_graph.append([name, name, 0])
    # bio_name = set(bio[:, 0]).union(bio[:, 1])
    # rea_name = set(reactome[:, 0]).union(set(reactome[:, 2])).union(bio_name)
    # print(len(rea_name & name_set))
    for line in bio:
        if line[0] in name_set and line[1] in name_set:
            result_graph.append([line[0], line[1], 6])
    for line in reactome:
        if line[0] in name_set and line[2] in name_set:
            if line[1] == "->":
                result_graph.append([line[0], line[2], 1])
                result_graph.append([line[2], line[0], 2])
            elif line[1] == "-|":
                result_graph.append([line[0], line[2], 4])
                result_graph.append([line[2], line[0], 5])
            else:
                result_graph.append([line[0], line[2], 3])
                result_graph.append([line[2], line[0], 3])
    # print(len(result_graph))
    return result_graph

def final_info(bp_list, cc_list, mf_list, graph, sl):
    graph_arr = np.array(graph)
    print("{:<5}{:<10}{:<10}".format("name", "nodes", "links"))
    print("{:<5}{:<10}".format("bp", len(bp_list)))
    print("{:<5}{:<10}".format("cc", len(cc_list)))
    print("{:<5}{:<10}".format("mf", len(mf_list)))
    print("{:<5}{:<10}{:<10}".format(
        "graph", len(set(graph_arr[:,0]).union(set(graph_arr[:,1]))), len(graph_arr)
    ))
    print("{:<5}{:<10}{:<10}".format(
        "SL", len(set(sl[:,0]).union(set(sl[:,1]))), len(sl)
    ))

def write_file(bp_list, cc_list, mf_list, graph, sl, bp_file, cc_file, mf_file, graph_file, sl_file):
    # bp_arr = np.array(bp_list)
    # cc_arr = np.array(cc_list)
    # mf_arr = np.array(mf_list)
    # print(bp_arr[:, 0])
    # print(cc_arr[:, 0])
    # print(mf_arr[:, 0])
    write_list(bp_file, bp_list)
    write_list(cc_file, cc_list)
    write_list(mf_file, mf_list)
    write_list(graph_file, graph)
    write_list(sl_file, sl)
    
def write_list(file, data):
    with open(file, 'w') as write_file:
        for line in data:
            for i in range(len(line)):
                if i == len(line) - 1:
                    write_file.write(str(line[i]))
                    write_file.write('\n')
                else:
                    write_file.write(str(line[i]))
                    write_file.write('\t')

if __name__ == "__main__":
    raw_path = "./data/raw"
    train_path = "./data/train"
    test_path = "./data/test"

    bi_file = "/BIOGRID-ORGANISM-Homo_sapiens-3.4.143.tab2.simple.tsv"
    bp_file = "/c5.go.bp.v7.3.symbols.pca64.pt"
    cc_file = "/c5.go.cc.v7.3.symbols.pca64.pt"
    mf_file = "/c5.go.mf.v7.3.symbols.pca64.pt"
    re_file = "/reactome-FI_links.tsv"
    na_file = "/HGNC_std_names.json"
    sl_file = "/Human_SL.csv"

    result_bp_file = "/bp.txt"
    result_cc_file = "/cc.txt"
    result_mf_file = "/mf.txt"
    result_gr_file = "/gra.txt"
    result_sl_file = "/sl.txt"

    '''
    load data
    '''
    bp = load_bp(raw_path + bp_file)
    cc = load_cc(raw_path + cc_file)
    mf = load_mf(raw_path + mf_file)
    std_name = load_na(raw_path + na_file)
    bio = load_bi(raw_path + bi_file)
    reactome = load_re(raw_path + re_file)
    sl = load_sl(raw_path + sl_file)

    basic_info(bp, cc, mf, bio, reactome, std_name, sl)

    bp, cc, mf, bio, reactome, std_name, sl = convert_name(bp, cc, mf, bio, reactome, std_name, sl)

    '''
    normalize features 
    '''
    bp = normalize_map(bp)
    cc = normalize_map(cc)
    mf = normalize_map(mf)

    basic_info(bp, cc, mf, bio, reactome, std_name, sl)
    '''
    append the missing feature
    '''
    bp_list, cc_list, mf_list = append_missing(bp, cc, mf, sl)

    '''
    merge bio and reactome and set rel_type
    '''
    graph = merge_graph(bio, reactome, sl)
    '''
    delete the name in sl
    '''

    '''
    show final info
    '''
    final_info(bp_list, cc_list, mf_list, graph, sl)

    '''
    write to file
    '''
    write_file(bp_list, cc_list, mf_list, graph, sl,
        train_path + result_bp_file, train_path + result_cc_file, train_path + result_mf_file,
        train_path + result_gr_file, train_path + result_sl_file)

