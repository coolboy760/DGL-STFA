'''
Author: chenzheng 2309712705@qq.com
Date: 2024-07-06 11:48:41
LastEditors: chenzheng 2309712705@qq.com
LastEditTime: 2024-07-07 13:51:07
FilePath: /DGL-STFA/model/dyg_regressor.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

from typing import Any, Mapping
import torch 
from torch_geometric.nn import GCNConv
from torch import nn
from torch import Tensor
from utils import *


class DGRLayer(nn.Module):

    def __init__(self,
                 cfg: Mapping[str, Any],
                 input_d: int) -> None:

        super().__init__()
        self.cfg = cfg
        self.gru = nn.GRU(input_d, cfg.gcn_d, cfg.n_gru_layers, batch_first=True)
        self.gcn = GCNConv(cfg.gcn_d, cfg.gcn_d, bias=False)
        
    def forward(self,
                gru_input: Tensor,
                edge_index_batch: Tensor,
                edge_attr_batch: Tensor,
                batch: Tensor) -> Tensor:

        gru_output, gru_hidden = self.gru(gru_input)
        B = gru_output.shape[0]
        gru_output = gru_output.reshape(gru_output.shape[0]*self.cfg.t_repetition, self.cfg.gcn_d)
        out = self.gcn(gru_output, edge_index_batch, edge_attr_batch)
        out = out.reshape(B, self.cfg.t_repetition, self.cfg.gcn_d)
        return out

class DygRegressor(nn.Module):

    def __init__(self,
                 cfg: Mapping[str, Any]) -> None:

        super().__init__()
        self.cfg = cfg
        self.dyg_regressor = nn.ModuleList([DGRLayer(cfg, cfg.gcn_d)])
        self.temporal_attention = TemporalAttention(cfg)
        self.fc = nn.Linear(cfg.gcn_d, cfg.n_steps, bias=True)
        
    def forward(self,
                node_features: Tensor,
                edge_index_batch: Tensor,
                edge_attr_batch: Tensor,
                batch: Tensor) -> Tensor:
    
        for dgc_layer in self.dyg_regressor:
            node_features = dgc_layer(node_features, edge_index_batch, edge_attr_batch, batch)
        x_temporal_attn = self.temporal_attention(node_features)
        node_features = node_features.reshape(node_features.shape[0]//self.cfg.n_neurons, self.cfg.t_repetition,self.cfg.n_neurons, self.cfg.gcn_d)
        out = x_temporal_attn * node_features
        out = torch.sum(out, (1, 2))
        out = out.view(node_features.shape[0], -1)
        out = self.fc(out)
        return out
    
class TemporalAttention(torch.nn.Module):

    def __init__(self,
                 cfg: Mapping[str, Any]) -> None:

        super().__init__()
        self.cfg = cfg
        T_ebd = int(cfg.tau*cfg.t_repetition)
        self.temporal_attn = torch.nn.Sequential(torch.nn.Linear(cfg.t_repetition, T_ebd, bias=False),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(T_ebd, cfg.t_repetition, bias=False),
                                     torch.nn.Sigmoid())
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]//self.cfg.n_neurons
        x_temporal_attn = x.view(batch_size, self.cfg.t_repetition, self.cfg.n_neurons*self.cfg.ebd_d)
        x_temporal_attn = torch.mean(x_temporal_attn, -1)
        x_temporal_attn = self.temporal_attn(x_temporal_attn)
        x_temporal_attn = x_temporal_attn.view(batch_size, self.cfg.t_repetition, 1, 1)
        return x_temporal_attn