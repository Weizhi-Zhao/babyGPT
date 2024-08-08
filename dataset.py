# it's said that np.memmap leaks memory when itering over the dataset
# TODO: detect the memory leak
# TODO: fix numpy memoryleak
# TODO: will torch mmap leak?
# TODO: compare torch and numpy performance

import torch
from torch.utils.data import Dataset
import numpy as np

class ShakespeareCharDataset(Dataset):
    def __init__(self, file_path, block_size):
        self.file_path = file_path
        self.block_size = block_size

    def __len__(self):
        data = np.memmap(self.file_path, dtype=np.uint16, mode='r')
        return len(data) - self.block_size
    
    def __getitem__(self, index):
        data = np.memmap(self.file_path, dtype=np.uint16, mode='r')
        x = torch.from_numpy(data[index:index+self.block_size].copy()).long()
        y = torch.from_numpy(data[index+1:index+self.block_size+1].copy()).long()
        return x, y
    

DATASETS = {
    "shakespeare_char": ShakespeareCharDataset
}