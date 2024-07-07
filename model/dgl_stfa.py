'''
Author: chenzheng 2309712705@qq.com
Date: 2024-07-06 11:48:41
LastEditors: chenzheng 2309712705@qq.com
LastEditTime: 2024-07-07 12:35:24
FilePath: /battery/DGL-STFA/model/dgl_stfa.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

from typing import Any, Mapping
import torch
import torch.nn as nn
from torch import Tensor
import torch_geometric as tg

from .dyg_preconstructor import DygPreconstructor
from .dyg_learner import DygLearner
from .dyg_regressor import DygRegressor
from utils import *

class DGLSTFA(nn.Module):

    def __init__(self,
                 cfg: Mapping[str, Any]) -> None:

        super().__init__()
        self.cfg = cfg
        self.dyg_proconstructor = DygPreconstructor(cfg)
        self.dyg_learner = DygLearner(cfg)
        self.dyg_regressor = DygRegressor(cfg)

    def forward(self, x: Tensor) -> Tensor:
        x_split = self.dyg_proconstructor(x)
        node_features, sparse_adjacency, edge_index_batch, edge_attr_batch, batch = self.dyg_learner(x_split)
        gru_input = node_features.transpose(1,2).reshape(x.shape[0]*self.cfg.n_neurons, self.cfg.t_repetition, -1)
        out = self.dyg_regressor(gru_input, edge_index_batch, edge_attr_batch, batch)
        return out,sparse_adjacency