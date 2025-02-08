import numpy as np 
import torch 
from MyNLPDataSet import MyNLPDataset
from torch.utils.data import DataLoader 
import gzip 

def cycle(loader):
    while True:
        for data in loader:
            yield data 

def get_loaders_enwiki8(seq_len, batch_size):
    # ---------prepare enwik8 data-----------
    with gzip.open('data/enwik8.gz') as file:
        data = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
        data_train, data_val = map(torch.from_numpy, np.split(data, [int(90e6)]))
    
    train_dataset = MyNLPDataset(data_train, seq_len)
    val_dataset = MyNLPDataset(data_val, seq_len)
    
    train_loader = cycle(DataLoader(train_dataset, batch_size=batch_size))
    val_loader = cycle(DataLoader(val_dataset, batch_size=batch_size))
    
    return train_loader, val_loader, val_dataset

def get_loaders_listops(seq_len, batch_size):
    # ---------prepare listops data-----------
    with gzip.open('data/lra_release.gz') as file:
        data = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
        data_train, data_val = map(torch.from_numpy, np.split(data, [int(90e6)]))
    
    train_dataset = MyNLPDataset(data_train, seq_len)
    val_dataset = MyNLPDataset(data_val, seq_len)
    
    train_loader = cycle(DataLoader(train_dataset, batch_size=batch_size))
    val_loader = cycle(DataLoader(val_dataset, batch_size=batch_size))
    
    return train_loader, val_loader, val_dataset