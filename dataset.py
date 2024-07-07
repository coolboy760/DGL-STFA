import numpy as np
import torch
import pickle
from torch.utils.data import Dataset


class BatteryDataset(Dataset):
    def __init__(self,data_x,data_y):
        self.data_x = data_x
        self.data_y = data_y
    def __getitem__(self, index):
        return torch.Tensor(self.data_x[index]),torch.Tensor(self.data_y[index])

    def __len__(self):
        return len(self.data_x)


class Normalization():
    def __init__(self,file_path=None):
        self.file_path = file_path
    def fit(self, train):
        if len(train.shape) == 1:
            self.min = min(train)
            self.max = max(train)
        elif len(train.shape) == 2:
            self.min = [min(train[i,:]) for i in range(train.shape[0])]
            self.max = [max(train[i,:]) for i in range(train.shape[0])]
        elif len(train.shape) == 3:
            self.min = [train[:,i,:].min() for i in range(train.shape[1])]
            self.max = [train[:,i,:].max() for i in range(train.shape[1])]
    
    def save(self,file_path):
        with open(file_path,'wb') as f:
            pickle.dump((self.max,self.min),f)
        f.close()

    def load(self,file_path):
        with open(file_path,'rb') as f:
            self.max,self.min = pickle.load(f)
        f.close()
    
    def normalize(self, a):
        data = a.copy()
        if len(data.shape) == 1:
            data = (data - self.min) / (self.max - self.min)
        elif len(data.shape) == 2:
            for i in range(data.shape[0]):
                data[i,:] = (data[i,:] - self.min[i]) / (self.max[i] - self.min[i])
        elif len(data.shape) == 3:
            for i in range(data.shape[1]):
                data[:,i,:] = (data[:,i,:] - self.min[i]) / (self.max[i] - self.min[i])
        return data

    def denormalize(self, data):
        a = data.copy()
        if len(a.shape) == 1:
            a = a * (self.max - self.min) + self.min
        elif len(a.shape) == 2:
            for i in range(a.shape[0]):
                a[i,:] = a[i,:] * (self.max[i] - self.min[i]) + self.min[i]
        elif len(a.shape) == 3:
            for i in range(a.shape[1]):
                a[:,i,:] = a[:,i,:] * (self.max[i] - self.min[i]) + self.min[i]
        return a
    
    def inverse_transform(self, y):
        
        if len(y.shape) == 1:
            y = y * (self.max - self.min) + self.min
        else:
            y = y * (self.max[-1] - self.min[-1]) + self.min[-1]
        return y
    
    def fit_and_normalize(self, train, test, val=None):
        self.fit(train)
        if val is not None:
            return self.normalize(train), self.normalize(test), self.normalize(val)
        else:
            return self.normalize(train), self.normalize(test)
        
def get_patch_from_item(index, item, seq_len):
    return item[:,index-seq_len:index]

def get_patchs_from_item(indexs, item, seq_len, pred_len):
    patchs = []
    pred = []
    for index in indexs:
        patchs.append(get_patch_from_item(index, item, seq_len))
        pred.append(item[-1,index+pred_len-1:index+pred_len])
    return np.array(patchs),np.array(pred)

def get_item_from_dataset(key, dataset):
    return dataset[key]

def get_patchs(indexs,key,dataset):
    item = get_item_from_dataset(key,dataset)
    patchs = get_patchs_from_item(indexs,item)
    return patchs