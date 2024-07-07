from typing import Any, Mapping
import torch 
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import os

def dict2obj(d):
    if isinstance(d, list):
        d = [dict2obj(x) for x in d]
    if not isinstance(d, dict):
        return d
    class C(object):
        pass
    o = C()
    for k in d:
        o.__dict__[k] = dict2obj(d[k])
    return o


def get_t_repetition(cfg: Mapping[str, Any]) -> torch.Tensor:
    t_repetition = (cfg.T - cfg.len_window)//cfg.stride+1
    return t_repetition


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(model, path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

def construct_graph(x_ebd: Tensor) -> Tensor:
    x_ebd = F.softmax(x_ebd, -1)
    adjacency_matrix = torch.matmul(x_ebd, torch.transpose(x_ebd, 2, 3))
    return adjacency_matrix

def get_coo(adjacency_matrix: Tensor) -> Tensor:
    matrix = adjacency_matrix > 0
    shape = matrix.shape
    n_nodes = shape[-1]
    if matrix.dim() == 4:
        matrix = matrix.reshape(shape[0]*shape[1],shape[2],shape[3])
    edge_indices = [torch.nonzero(matrix[i], as_tuple=False)+i*n_nodes for i in range(matrix.shape[0])]
    edge_index_batch = torch.concat(edge_indices,0).T
    edge_attr_batch = adjacency_matrix[adjacency_matrix > 0].unsqueeze(-1)
    batch = edge_index_batch[1]
    return edge_index_batch, edge_attr_batch, batch
