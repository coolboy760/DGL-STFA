from typing import Any, Mapping
import torch
from torch import nn
from torch import Tensor
from utils import *

class DygLearner(nn.Module):

    def __init__(self,
                 cfg: Mapping[str, Any]) -> None:

        super().__init__()
        self.ebd_region = RegionEmbedder(cfg)
        self.spatial_attention = SpatialAttention(cfg)
        self.sparsify = Sparsify(cfg)
        
    def forward(self, x_split: Tensor) -> Tensor:
        x_ebd = self.ebd_region(x_split)
        node_features =  x_ebd.clone()
        x_spatial_attention = self.spatial_attention(x_ebd)
        x_ebd = x_spatial_attention * x_ebd
        adjacency_matrix = construct_graph(x_ebd)
        sparse_adjacency = self.sparsify(adjacency_matrix)
        edge_index_batch, edge_attr_batch, batch = get_coo(sparse_adjacency)
        return node_features, sparse_adjacency, edge_index_batch, edge_attr_batch, batch

class MSTC(nn.Module):
    def __init__(self,
                 cfg: Mapping[str, Any],
                 dilation: int) -> None:

        super().__init__()

        self.cfg = cfg
        self.dilation = dilation
        self.t_conv_0 = nn.Conv1d(cfg.itcn_d, cfg.itcn_d//3, cfg.kernel_list[0], dilation=dilation, padding=(cfg.kernel_list[0]-1)*dilation)
        self.t_conv_1 = nn.Conv1d(cfg.itcn_d, cfg.itcn_d//3, cfg.kernel_list[1], dilation=dilation, padding=(cfg.kernel_list[1]-1)*dilation)
        self.t_conv_2 = nn.Conv1d(cfg.itcn_d, cfg.itcn_d//3, cfg.kernel_list[2], dilation=dilation, padding=(cfg.kernel_list[2]-1)*dilation)
   

    def clip_end(self, x: Tensor, i: int) -> Tensor:
        padding = (self.cfg.kernel_list[i]-1)*self.dilation
        x = x[:, :, :-padding].contiguous()
        return x

    def forward(self, x_split: Tensor) -> Tensor:
        x_cat = [self.clip_end(self.t_conv_0(x_split), 0), self.clip_end(self.t_conv_1(x_split), 1), self.clip_end(self.t_conv_2(x_split), 2)]
        x_cat = torch.cat(x_cat, 1)
        x_out = torch.relu(x_cat)
        return x_out
        
class MSTCN(nn.Module):
    def __init__(self,
                 cfg: Mapping[str, Any],
                 n_layers: int) -> None:

        super().__init__()
        self.cfg = cfg
        self.core = nn.Sequential(*[MSTC(cfg, 2**i) for i in range(n_layers)])

    def forward(self, x_split: Tensor) -> Tensor:
        batch_size = x_split.shape[0]
        x_split = x_split.transpose(2,3).reshape(batch_size*self.cfg.n_neurons, self.cfg.itcn_d, self.cfg.t_repetition)
        x_split = self.core(x_split)
        x_split = x_split.reshape(batch_size, self.cfg.n_neurons, self.cfg.itcn_d,self.cfg.t_repetition).permute(0,3,1,2)
        return x_split

class RegionEmbedder(nn.Module):
    def __init__(self,
                 cfg: Mapping[str, Any]) -> None:

        super().__init__()
        self.input_layer = nn.Linear(cfg.len_window, cfg.itcn_d)
        self.dilated_inception = MSTCN(cfg, cfg.n_itcn_layers)
        self.output_fc = nn.Sequential(nn.Linear(cfg.itcn_d, cfg.itcn_d), nn.ReLU(), nn.Linear(cfg.itcn_d, cfg.ebd_d))
        
    def forward(self, x_split: Tensor) -> Tensor:
        x_split = self.input_layer(x_split)
        x_split = self.dilated_inception(x_split)
        x_split = self.output_fc(x_split) 
        return x_split
    
    
class SpatialAttention(nn.Module):

    def __init__(self,
                 cfg: Mapping[str, Any]) -> None:

        super().__init__()
        n_neurons_ebd = int(cfg.tau*cfg.n_neurons)
        self.spatial_attn = nn.Sequential(nn.Linear(cfg.n_neurons, n_neurons_ebd, bias=False),
                                          nn.ReLU(),
                                          nn.Linear(n_neurons_ebd, cfg.n_neurons, bias=False),
                                          nn.Sigmoid())
        
    def forward(self, x_ebd: Tensor) -> Tensor:
        x_spatial_attn = torch.mean(x_ebd, -1)
        x_spatial_attn = self.spatial_attn(x_spatial_attn)
        x_spatial_attn = x_spatial_attn.unsqueeze(-1)
        return x_spatial_attn
    
class Sparsify(nn.Module):

    def __init__(self,
                 cfg: Mapping[str, Any]) -> None:

        super().__init__()
        self.threshold = nn.parameter.Parameter(torch.full((1,), -6.0))

    def forward(self, adjacency_matrix: Tensor) -> Tensor:
        sparse_adjacency = torch.relu(adjacency_matrix - torch.sigmoid(self.threshold))
        return sparse_adjacency