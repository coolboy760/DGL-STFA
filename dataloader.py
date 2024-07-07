'''
Author: chenzheng 2309712705@qq.com
Date: 2024-07-07 04:03:34
LastEditors: chenzheng 2309712705@qq.com
LastEditTime: 2024-07-07 13:47:51
FilePath: /battery/DGL-STFA/dataloader.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import os
import numpy as np
import pickle
from torch.utils.data import DataLoader
import warnings
from dataset import *

warnings.filterwarnings('ignore')

def load_dataset(dataset_dir,size,val,test,batch_size,valid_batch_size= None, test_batch_size=None):
    seq_len = size[0]
    label_len = size[1]
    pred_len = size[2]
    with open(os.path.join(dataset_dir,'feature.pkl'),'rb') as f:
        data_raw = pickle.load(f)
    f.close()
    keys = list(data_raw.keys())
    train = keys.copy()
    data = {}
    train_data = []
    for key in train:
        train_data.append(data_raw[key])
    train_data = np.concatenate(train_data,axis=1)
    scaler = Normalization()
    scaler.fit(train_data)
    scaler.save(os.path.join(dataset_dir,'scaler.pkl'))
    for key in val:
        if key in train:
            train.remove(key)
    for key in test:
        if key in train:
            train.remove(key)
    for category in ['train', 'val', 'test']:
        data_x = []
        data_y = []
        for key in locals()[category]:
            CYCLE_NUM = len(data_raw[key][0])
            item = data_raw[key]
            # print(item.shape)
            item = scaler.normalize(item)
            for index in range(CYCLE_NUM- seq_len - pred_len):
                # s_begin = index+ PREV_NUMok
                s_begin = index
                s_end = index + seq_len 
                r_begin = s_end 
                r_end = r_begin + pred_len
                seq_x = item[:,s_begin:s_end]
                seq_y = item[-1,r_end-1:r_end]
                data_x.append(seq_x)
                data_y.append(seq_y)
        data['x_' + category] = np.array(data_x)
        data['y_' + category] = np.array(data_y)
    train_loader = DataLoader(BatteryDataset(data['x_train'], data['y_train']), batch_size,shuffle=True,num_workers=8)
    val_loader = DataLoader(BatteryDataset(data['x_val'], data['y_val']), valid_batch_size,num_workers=8)
    test_loader = DataLoader(BatteryDataset(data['x_test'], data['y_test']), test_batch_size,num_workers=8)
    return train_loader,val_loader,test_loader,scaler

