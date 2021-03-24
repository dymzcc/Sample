import dgl
import torch as th
import numpy as np


class NeighborSampler(object):
    def __init__(self, g, fanouts):
        """
        fanouts 为采样节点的数量。
        """
        self.g = g
        self.fanouts = fanouts

    def sample_blocks(self,seeds):
        seeds = th.LongTensor(np.asarray(seeds))
        blocks = []
        for fanout in self.fanouts:

            frontier = dgl.sampling.sample_neighbors(self.g, seeds, fanout, prob='prob', replace=True)
            block = dgl.to_block(frontier, seeds)
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return blocks

