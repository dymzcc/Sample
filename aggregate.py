import torch.nn as nn
import torch as th
import torch.nn.functional as F
import dgl
import torch.optim as optim
import torch.multiprocessing as mp
from dgl.utils import expand_as_pair, check_eq_shape
from torch.utils.data import DataLoader
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import time
import argparse


class SAGEConv(nn.Module):
    def __init__(self,
                 in_feats,          #输入特征维数
                 out_feats,         #输出特征维数
                 aggregator_type,
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=None):
        super(SAGEConv, self).__init__()

        # expand_as_pair 函数可以返回一个二维元组。
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        # aggregator type: mean/pool/lstm/gcn
        if aggregator_type == 'pool':
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == 'lstm':
            self.lstm = nn.LSTM(self._in_src_feats, self._in_src_feats, batch_first=True)
        if aggregator_type != 'gcn':
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):

        gain = nn.init.calculate_gain('relu')
        if self._aggre_type == 'pool':
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == 'lstm':
            self.lstm.reset_parameters()
        if self._aggre_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
        return self.fc_neigh.weight

    def _lstm_reducer(self, nodes):

        m = nodes.mailbox['m'] # (B, L, D)
        batch_size = m.shape[0]
        h = (m.new_zeros((1, batch_size, self._in_src_feats)),
             m.new_zeros((1, batch_size, self._in_src_feats)))
        _, (rst, _) = self.lstm(m, h)
        return {'neigh': rst.squeeze(0)}

    def forward(self, graph, feat):

        graph = graph.local_var()

        if isinstance(feat, tuple):
            feat_src = self.feat_drop(feat[0])
            feat_dst = self.feat_drop(feat[1])
        else:
            feat_src = feat_dst = self.feat_drop(feat)

        h_self = feat_dst    #目标节点的嵌入向量

        if self._aggre_type == 'mean':
            graph.srcdata['h'] = feat_src     #每个节点关联上其特征
            graph.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']    #邻域节点的聚合向量
        elif self._aggre_type == 'gcn':

            check_eq_shape(feat)
            graph.srcdata['h'] = feat_src
            graph.dstdata['h'] = feat_dst     # same as above if homogeneous
            graph.update_all(fn.copy_src('h', 'm'), fn.sum('m', 'neigh'))
            degs = graph.in_degrees().to(feat_dst)
            h_neigh = (graph.dstdata['neigh'] + graph.dstdata['h']) / (degs.unsqueeze(-1) + 1)
        elif self._aggre_type == 'pool':
            graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
            graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']
        elif self._aggre_type == 'lstm':
            graph.srcdata['h'] = feat_src
            graph.update_all(fn.copy_src('h', 'm'), self._lstm_reducer)
            h_neigh = graph.dstdata['neigh']
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

        # GraphSAGE GCN does not require fc_self.
        if self._aggre_type == 'gcn':
            rst = self.fc_neigh(h_neigh)
        else:
            rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)   #目标节点向量和邻域向量拼接
        # activation
        if self.activation is not None:
            rst = self.activation(rst)
        # normalization
        if self.norm is not None:
            rst = self.norm(rst)
        return rst
    

# g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
# g = dgl.add_self_loop(g)
# feat = th.ones(6, 10)
# conv = SAGEConv(10, 2, 'mean')
# res = conv(g, feat)
# print(feat)
# weight = conv.reset_parameters()
# print(weight)

