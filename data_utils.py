import numpy as np
from tqdm import tqdm
import json
import random
from shutil import copyfile


def load_name(std_name):
    with open(std_name, 'r') as read_file:
        name_map = json.load(read_file)
    return name_map

def process_name(name_map, GTEx_file, GO_file, train_source):
    GTEx_array = np.loadtxt(GTEx_file, dtype=str, delimiter='\t', skiprows=3)[:, 1:]
    with open(GO_file, 'r') as read_file: 
        GO = json.load(read_file)
    
    GO_map = GO['gene2id']
    GO_list = []
    except_GO = 0
    stdGO_map = {}
    for key in GO_map.keys():
        if key in name_map.keys():
            GO_list.append([name_map[key]] + GO['embedding'][GO_map[key]])
        else:
            except_GO += 1
            GO_list.append([key] + GO['embedding'][GO_map[key]])
    GO_array = np.array(GO_list)
    GO_data = normalize(GO_array[:, 1:].astype(float))

    except_GTEx = 0
    for line in GTEx_array:
        if line[0] in name_map.keys():
            line[0] = name_map[line[0]]
        else: except_GTEx += 1
    GTEx_data = normalize(GTEx_array[:, 1:].astype(float))

    print(f"GO except {except_GO} GTEx execpt {except_GTEx}")

    return GO_array[:, 0], GO_data, GTEx_array[:, 0], GTEx_data


def load_train(name_map, source_file):
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
    train_list = []
    except_num = 0
    with open(source_file, 'r') as readfile:
        for line in readfile.readlines():
            tmp = line.split('\t')
            if tmp[-2] not in useless_method:
                if tmp[0] in name_map.keys() and tmp[2] in name_map.keys():
                    train_list.append([name_map[tmp[0]], name_map[tmp[2]]])
                else:
                    except_num += 1
                    train_list.append([tmp[0], tmp[2]])
    print(f"train except {except_num}")
    return np.array(train_list)

def load_links(links_file):
    links = np.loadtxt(links_file, dtype=str, delimiter='\t')
    return links


def cut_data(train_data, GO_feat, GTEx_feat, links):
    train_set = set(train_array[:, 0]).union(set(train_array[:, 1]))
    GO_result = []
    GTEx_result = []
    links_result = []
    for line in GO_feat:
        if line[0] in train_set: GO_result.append(line)
    for line in GTEx_feat:
        if line[0] in train_set: GTEx_result.append(line)
    for line in links:
        if line[0] in train_set and line[2] in train_set:
            links_result.append(line)

    # append the missing data to feature
    return np.array(GO_result), np.array(GTEx_result), np.array(links_result)


def normalize(unnorm):
    normed = np.log(unnorm + 1)
    normed = (normed - normed.mean(axis=0)) / normed.std(axis=0)
    return np.nan_to_num(normed)

def test(train_array, tmp_file, name_map):
    train_set = set(train_array[:, 0]).union(set(train_array[:, 1]))
    gene_arr = np.loadtxt(tmp_file, dtype=str, delimiter='\t')
    gene_set = set()
    for i in gene_arr[:, 1]:
        if i in name_map.keys(): gene_set.add(name_map[i])
        else: gene_set.add(i)
    for i in gene_arr[:, 4]:
        if i in name_map.keys(): gene_set.add(name_map[i])
        else: gene_set.add(i)

    inter = gene_set & train_set
    print(f"inter {len(inter)}")
