from sklearn.model_selection import train_test_split
import torch.optim as optim
#from dataload import load_lfr_1,get_feature_lfr_1,get_label_lfr_1,threshold,egde_pro
from dataload_2 import load_amazon,get_label_amazon,get_feature_amazon,egde_pro,threshold
from aggregate import SAGEConv
import dgl
import torch as th
import torch.nn as nn
from neighbor_sample import NeighborSampler
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import scipy.sparse as sp
import time
import dgl.nn.pytorch as dglnn
import tqdm
from nmi import onmi
from sklearn.metrics import f1_score


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):

        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_dst = h[:block.number_of_dst_nodes()]
            h = layer(block, (h, h_dst))
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, batch_size, device):

        nodes = th.arange(g.number_of_nodes())
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.number_of_nodes(),
                         self.n_hidden if l != len(self.layers) - 1 else self.n_classes)
            for start in tqdm.trange(0, len(nodes), batch_size):
                end = start + batch_size
                batch_nodes = nodes[start:end]
                block = dgl.to_block(dgl.in_subgraph(g, batch_nodes), batch_nodes)
                input_nodes = block.srcdata[dgl.NID]
                h = x[input_nodes].to(device)
                h_dst = h[:block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)
                y[start:end] = h.cpu()
            x = y
        return y

def compute_acc(pred, label,g):
    """
    计算准确率
    """
    thr = threshold(g)
    print(thr)
    for i in range(len(pred)):
        for j in range(len(pred[0])):
            if(pred[i][j] > 4*thr):pred[i][j] = 1
            else: pred[i][j] = 0

    pre_com = []
    for j in range(len(pred[0])):
        com = []
        for i in range(len(pred)):
            if (pred[i][j] == 1):
                com.append(i)
        pre_com.append(set(com))

    lab_com = []
    for j in range(len(label[0])):
        com = []
        for i in range(len(label)):
            if (label[i][j] == 1):
                com.append(i)
        lab_com.append(set(com))

    nmi = onmi(pre_com, lab_com)

    emb = pred
    embed_m = sp.csr_matrix(emb.T, dtype=np.uint32)

    label_m = sp.csr_matrix(label.T, dtype=np.uint32)

    n = (label_m.dot(embed_m.T)).toarray().astype(float)  # cg * cd
    p = n / np.array(embed_m.sum(axis=1)).clip(min=1).reshape(-1)
    r = n / np.array(label_m.sum(axis=1)).clip(min=1).reshape(-1, 1)
    f1 = 2 * p * r / (p + r).clip(min=1e-10)
    f1_s1 = f1.max(axis=1).mean()
    f1_s2 = f1.max(axis=0).mean()
    f1_s = (f1_s1 + f1_s2) / 2
    # f1_s = f1_score(label,pred,average="micro")
    return f1_s,nmi

def evaluate(model, g, inputs, labels, val_mask, batch_size, device):
    """
    评估模型，调用 model 的 inference 函数
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, inputs, batch_size, device)
    model.train()
    return (pred)
    #return compute_acc(pred[val_mask], labels[val_mask])

def load_subtensor(g, labels, seeds, input_nodes, device):
    """
    将一组节点的特征和标签复制到 GPU 上。
    """
    batch_inputs = g.ndata['feature'][input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels

if __name__ == '__main__':
    device = th.device('cpu')
    num_epochs = 1000
    num_hidden = 100
    num_layers = 2
    fan_out = '8,25'
    batch_size = 1000
    log_every = 20  # 记录日志的频率
    eval_every = 5
    lr = 0.01
    dropout = 0

    g = dgl.DGLGraph()
    n_classes = 100
    g = dgl.from_networkx(load_amazon())

    feature = th.Tensor(get_feature_amazon())
    in_feats = feature.shape[1]
    label = th.Tensor(get_label_amazon())
    weight = th.Tensor(egde_pro())
    g.ndata['feature'] = feature
    g.ndata['label'] = label
    g.edata['pro'] = weight

    X_train, X_text = train_test_split(g.nodes(), test_size=0.9)

    # 采样器
    sampler = NeighborSampler(g, [int(fanout) for fanout in fan_out.split(',')])

    dataloader = DataLoader(
        dataset=X_train.numpy(),
        batch_size=batch_size,
        collate_fn=sampler.sample_blocks,
        shuffle=True,
        drop_last=False,
        num_workers=0)

    #建立模型和优化器
    #model = GraphSAGE(in_feats, num_hidden, n_classes, num_layers, F.relu, dropout)
    model = GraphSAGE(in_feats, num_hidden, n_classes, num_layers, F.logsigmoid,dropout)
    model = model.to(device)
    loss_fcn = nn.BCELoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    #训练

    for epoch in range(num_epochs):
        tic = time.time()

        for step, blocks in enumerate(dataloader):
            tic_step = time.time()

            input_nodes = blocks[0].srcdata[dgl.NID]
            seeds = blocks[-1].dstdata[dgl.NID]

            # Load the input features as well as output labels
            batch_inputs, batch_labels = load_subtensor(g, label, seeds, input_nodes, device)

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            m = nn.Sigmoid()
            loss = loss_fcn(m(batch_pred), batch_labels)
            #print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    eval_acc_1 = evaluate(model,g,g.ndata['feature'],label,X_text,batch_size, device)
    amazon_f1, amazon_nmi = compute_acc(eval_acc_1,label,load_amazon())

    # g2 = dgl.DGLGraph()
    # g2 = dgl.from_networkx(load_lfr_2())
    # feature2 = th.Tensor(get_feature_lfr_2())
    # in_feats2 = feature2.shape[1]
    # label2 = th.Tensor(get_label_lfr_2())
    # g2.ndata['feature'] = feature2
    # g2.ndata['label'] = label2
    # g2_train = g2.nodes()
    #
    # eval_acc_2 = evaluate(model,g2,g2.ndata['feature'],label2,g2_train,batch_size, device)
    # lfr_2_f1 = compute_acc(eval_acc_2, label2, load_lfr_2())

    print(amazon_f1)
    print(amazon_nmi)



