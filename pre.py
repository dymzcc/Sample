import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter
from networkx.algorithms.community import asyn_fluidc

def load_graph():
    a = pd.read_table('/Users/fartinhands/Desktop/GDC/Dataset/dataset/LFR_dataset/two/1/network.txt', sep='\s+',
                      header=None)
    x = a[0].values-1
    y = a[1].values-1
    g = nx.Graph()
    for n in range(0, len(x)):
        g.add_node(x[n])
        g.add_edge(x[n], y[n])
    return g

def get_label():
    b = '/Users/fartinhands/Desktop/GDC/Dataset/dataset/LFR_dataset/two/1/community.txt'
    data = []
    with open(b, 'r') as f:
        line = f.readline()
        while line:
            eachline = line.split()
            read_data = [int(x) for x in eachline[0:]]
            data.append(read_data)
            line = f.readline()
    label = np.zeros((2000, 18))
    for i in range(len(data)):
        for j in range(1, len(data[i])):
            label[i][data[i][j]-1] = 1
    return label,data

def feature_pre_1():        #用infomap初始化

    a = pd.read_table('/Users/fartinhands/Desktop/GDC/Dataset/dataset/LFR_dataset/one/2/init_feature.txt', sep='\s+',
                      header=None)
    x = list(a[0].values)
    y = list(a[1].values)
    z = list(zip(x,y))
    z.sort(key=(lambda x:x[0]))

    com_list = []
    for i in range(18):
        com = []
        for j in range(len(z)):
            if(z[j][1]==(i+1)):
                com.append(z[j][0])
        com_list.append(com)
    print(com_list)

    laber,data = get_label()

    cor = []           #建立infomap发现的社区和真实社区的大致对应关系
    for i in range(18):
        corr = []
        for n in com_list[i]:
            x = []
            x.append(data[n-1][1])
        y = max(Counter(x))
        cor.append([i+1,y])
    print(cor)

    feature = []
    for i in range(18):
        j = cor[i][1]
        for n in com_list[i]:
            feature.append([n,j])

    file = open('/Users/fartinhands/Desktop/GDC/Dataset/dataset/LFR_dataset/one/1/feature.txt','w')

    for i in range(len(feature)):
        for j in range(len(feature[i])):
            file.write(str(feature[i][j]))  # write函数不能写int类型的参数，所以使用str()转化
            file.write('\t')  # 相当于Tab一下，换一个单元格
        file.write('\n')  # 写完一行立马换行
    file.close()

def get_feature_2():         #用asyn_fluidc初始化
    g = load_graph()
    comt = list(asyn_fluidc(g, 18, 100))
    feature = []
    for i in range(len(comt)):
        comt[i] = list(comt[i])
        for j in range(len(comt[i])):
            feature.append([comt[i][j],i+1])

    file = open('/Users/fartinhands/Desktop/GDC/Dataset/dataset/LFR_dataset/two/1/init_feature.txt', 'w')

    for i in range(len(feature)):
        for j in range(len(feature[i])):
            file.write(str(feature[i][j]))  # write函数不能写int类型的参数，所以使用str()转化
            file.write('\t')  # 相当于Tab一下，换一个单元格
        file.write('\n')  # 写完一行立马换行
    file.close()

feature_pre_1()