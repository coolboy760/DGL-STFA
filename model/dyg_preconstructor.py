from typing import Any, Mapping
import torch
from torch import nn
from torch import Tensor

class DygPreconstructor(nn.Module):

    def __init__(self,
                 cfg: Mapping[str, Any]) -> None:

        super().__init__()
        self.cfg = cfg
        
    def forward(self, x: Tensor) -> Tensor:
        cfg = self.cfg
        x_split = torch.stack([x[:, :, t*cfg.stride:t*cfg.stride+cfg.len_window] for t in range(cfg.t_repetition)], 2)
        return x_split
