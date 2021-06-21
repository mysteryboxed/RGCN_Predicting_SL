import numpy as np
import json
import random
from shutil import copyfile
from utils_ import *
from multiprocessing import Pool
from functools import partial
#import torch
#from pyrwr.ppr import PPR
#from tqdm import tqdm

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
#gene number:10219
name = np.loadtxt(train_path + bp_file, dtype=str, delimiter='\t')[:,0]
name.astype(int)
links = np.loadtxt(train_path + sl_file, dtype=int)
temp=np.ones([1,2],dtype=int)
temp[:,0]=1930
temp[:,1]=7345
print(temp in links)
print(links[0][:])
#id2name, name2id = idMap(train_path + bp_file)
#ppi_graph = constructGraph(train_path + gr_file, name2id)[0]
sl_n=np.empty((0,2))
i=0
while True:
    temp = np.random.randint(0,10210,size=(1,2))
    if(temp in links):
        #print('skip')
        continue
    else:
        #print('add')
        #np.append(sl_n,temp) #考虑如何添加数组的问题
        i+=1
    if(i==109):#测试用，真正使用时请改成36746
        break
print(sl_n)
print(sl_n.shape)
