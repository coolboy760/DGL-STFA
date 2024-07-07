'''
Author: chenzheng 2309712705@qq.com
Date: 2023-08-28 16:15:12
LastEditors: chenzheng 2309712705@qq.com
LastEditTime: 2024-07-07 13:31:56
FilePath: /battery/dynamic_graph_structure_learning/trainer.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
from typing import Any, List, Mapping
import numpy as np
import torch
import time
import torch.nn.functional as F
from torch.optim import lr_scheduler 
from model.dgl_stfa import DGLSTFA
from utils import get_t_repetition,EarlyStopping
from dataloader import load_dataset


class Trainer():

    def __init__(self,
                 cfg: Mapping[str, Any],test = False) -> None:

        self.cfg = cfg
        self.get_model(cfg)
        if test and os.path.exists(cfg.state_dict_path):
            self.model.load_state_dict(torch.load(cfg.state_dict_path+'/checkpoint.pth', map_location=f'{cfg.device}'))
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.device = torch.device(f'cuda:{cfg.device_idx}' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def get_model(self,cfg):
        cfg.t_repetition = get_t_repetition(cfg)
        self.model = DGLSTFA(cfg)


    def train_step(self,train_loader):

        time_now = time.time()
        train_loss = []
        iter_count = 0
        for i, (batch_x, batch_y) in enumerate(train_loader):
            iter_count += 1
            self.optimizer.zero_grad()
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            out, adj = self.model(batch_x)
            loss = F.mse_loss(out, batch_y)+ self.cfg.lamda * torch.norm(adj, p=1) 
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())
            if (i + 1) % 100 == 0:
                print("\titers: {0}| loss: {1:.7f}".format(i + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                print('\tspeed: {:.4f}s/iter'.format(speed))
                iter_count = 0
                time_now = time.time()
        train_loss = np.average(train_loss)  
        return train_loss

    def train(self,train_loader, val_loader) -> Mapping[str, Any]:
        self.model.train()
        train_steps = len(train_loader)
        scheduler = lr_scheduler.OneCycleLR(optimizer = self.optimizer ,
                                    steps_per_epoch = train_steps,
                                    pct_start = self.cfg.pct_start,
                                    epochs = self.cfg.n_episodes,
                                    max_lr = self.cfg.lr)
        early_stopping = EarlyStopping(patience=self.cfg.patience, verbose=True)
        for epoch in range(self.cfg.n_episodes):
            train_loss = self.train_step(train_loader)
            if epoch % 20 == 0:
                pass
            vali_loss = self.test(val_loader)
            print("Epoch: {0}| Train Loss: {1:.7f} Vali Loss: {2:.7f} ".format(
                epoch + 1, train_loss, vali_loss))
            early_stopping(vali_loss, self.model.state_dict(), self.cfg.state_dict_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            scheduler.step()
    
    def test(self,data_loader):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(data_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                out,_ = self.model(batch_x)
                loss = F.mse_loss(out, batch_y)
                losses.append(loss.item())
        avg_loss = np.average(losses)
        self.model.train()
        return avg_loss
    
    def predict(self,sample):
        self.model.eval()
        with torch.no_grad():
            out,adj_matrix = self.model(sample)
        return out,adj_matrix