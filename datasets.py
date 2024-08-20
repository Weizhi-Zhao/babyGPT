# it's said that np.memmap leaks memory when itering over the dataset
# detect the memory leak? there is no leak
# will torch mmap leak? no and torch is much faster

from PIL import Image
from pathlib import Path
from tokenizer import Tokenizer
from torch.utils.data import Dataset
from torchvision import transforms
import pickle
import random
import torch
import json


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


class ImageCaptionDataset(Dataset):
    """
    data: list[dict[img_path, caption]]
    the caption must has at least 1 tokens
    """
    def __init__(self, data_path: Path, block_size, tokenizer: Tokenizer):
        self.data_path = data_path
        # self.data = torch.load(self.data_path, mmap=True)
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = self.data_path.parent / self.data[index]['img']
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        caption = self.data[index]['caption']
        tokens = self.tokenizer.encode(caption)
        tokens_len = len(tokens)
        # [0, 1, 2, 3, 4, 5]
        if tokens_len <= self.block_size + 1:
            x_start = 0
            y_end = tokens_len
        else:
            x_start = random.randint(0, tokens_len - self.block_size - 1)
            y_end = x_start + self.block_size + 1
        if y_end - x_start - 1 < self.block_size:
            x_pad = torch.ones(self.block_size - (y_end - 1 - x_start), dtype=torch.long)
            y_pad = -torch.ones(self.block_size - (y_end - 1 - x_start), dtype=torch.long)
            x = torch.cat((torch.tensor(tokens[x_start:y_end-1], dtype=torch.long), x_pad))
            y = torch.cat((torch.tensor(tokens[x_start+1:y_end], dtype=torch.long), y_pad))
        elif y_end - x_start - 1 == self.block_size:
            x = torch.tensor(tokens[x_start:y_end-1], dtype=torch.long)
            y = torch.tensor(tokens[x_start+1:y_end], dtype=torch.long)
        else:
            raise ValueError("end - start > self.block_size")
        assert x.size(0) == self.block_size, f"x.size(0) = {x.size(0)}, self.block_size = {self.block_size}"
        assert y.size(0) == self.block_size, f"y.size(0) = {y.size(0)}, self.block_size = {self.block_size}"
        return img, x, y


class PretrainDataset(Dataset):
    def __init__(self, data_path, block_size):
        self.data = torch.load(data_path, mmap=True, weights_only=True)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, index):
        x = self.data[index:index+self.block_size]
        y = self.data[index+1:index+self.block_size+1]
        return x, y


DATASETS = {
    "shakespeare_char": ShakespeareCharDataset,
    "shakespeare": ShakespeareDataset,
    "chinese_poetry": ChinesePoetryDataset,
    "image_caption": ImageCaptionDataset
}