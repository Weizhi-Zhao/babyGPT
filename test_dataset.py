import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
from torch.utils.data import DataLoader, RandomSampler
from datasets import DATASETS, ShakespeareCharDataset
import time
from tqdm import tqdm
import argparse
import tracemalloc
from loguru import logger
import sys


if __name__ == '__main__':
    logger.remove()
    logger.add(sys.stderr, format="{message}", level="INFO")


    torch.manual_seed(321)
    torch.cuda.manual_seed(321)
    torch.cuda.reset_peak_memory_stats()
    # tracemalloc.start()

    dataset_class = ShakespeareCharDataset
    dataset = dataset_class('data/shakespeare_char/train.pt', 256)
    sampler = RandomSampler(dataset, replacement=True, num_samples=100000)
    dataloader = DataLoader(dataset, 
                            batch_size=100,
                            sampler=sampler,)

    start = time.perf_counter()
    for x, y in tqdm(dataloader):
        x.to('cuda')
        y.to('cuda')
    logger.info(f"Time: {time.perf_counter() - start:.3f}s")
    # logger.info(f"cpu memory: {tracemalloc.get_traced_memory()[1] / 1024**2:.3f} MB")
    logger.info(f"gpu memory: {torch.cuda.max_memory_allocated() / 1024**2:.3f} MB")
