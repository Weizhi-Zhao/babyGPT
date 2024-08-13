# it's said that np.memmap leaks memory when itering over the dataset
# detect the memory leak? there is no leak
# will torch mmap leak? no and torch is much faster

import torch
from torch.utils.data import Dataset
import pickle
from pathlib import Path


class ShakespeareCharDataset(Dataset):
    def __init__(self, file_path, block_size):
        self.file_path = file_path
        self.block_size = block_size
        self.data = torch.load(self.file_path, weights_only=True, mmap=True)

    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, index):
        x = self.data[index:index+self.block_size].long()
        y = self.data[index+1:index+self.block_size+1].long()
        return x, y
    

class ShakespeareDataset(Dataset):
    def __init__(self, file_path, block_size):
        self.file_path = file_path
        self.block_size = block_size
        self.data = torch.load(self.file_path, weights_only=True, mmap=True)

    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, index):
        x = self.data[index:index+self.block_size]
        y = self.data[index+1:index+self.block_size+1]
        return x, y


class ChinesePoetryDataset(Dataset):
    def __init__(self, file_path, block_size):
        self.file_path = file_path
        self.block_size = block_size
        self.data = torch.load(self.file_path, weights_only=True, mmap=True)

        meta_path = Path(file_path).parent / 'meta.pkl'
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        ctoi = meta['ctoi']
        self.end_token = ctoi['e']

    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, index):
        x = self.data[index:index+self.block_size]
        y: torch.Tensor = self.data[index+1:index+self.block_size+1]
        if (end_pos := (y == self.end_token)).any().item():
            # slicing returns a view, so we need to copy the tensor
            y = y.clone()
            end_idx = end_pos.nonzero()[0].item()
            y[end_idx+1:] = -1
        return x, y


DATASETS = {
    "shakespeare_char": ShakespeareCharDataset,
    "shakespeare": ShakespeareDataset,
    "chinese_poetry": ChinesePoetryDataset,
}