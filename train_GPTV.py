import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
from datasets import ImageCaptionDataset
from tokenizer import Tokenizer
from loguru import logger
from matplotlib import pyplot as plt
from models import GPTV
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from utils import load_config, generate_one_sequence, save_loss_fig, save_checkpoint
import argparse
import math
import sys
import time
import torch
from omegaconf import OmegaConf, DictConfig
from pathlib import Path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')


def get_dataset(cfg):
    train_set = ImageCaptionDataset(
        Path(cfg.train_set_path), cfg.block_size, Tokenizer(cfg.meta_path)
    )
    return train_set


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad
def estimate_loss(model, train_set, cfg, eval_num_samples=None):
    model.eval()
    out = {}

    if eval_num_samples is None:
        eval_num_samples = cfg.eval_num_samples * cfg.batch_size

    eval_train_sampler = RandomSampler(
        train_set, replacement=False, num_samples=eval_num_samples)
    eval_train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, sampler=eval_train_sampler)
    
    losses = []
    for img, x, y in eval_train_loader:
        img = img.to(device)
        x = x.to(device)
        y = y.to(device)
        logits, loss = model(img, x, y)
        losses.append(loss.item())
    out['train'] = sum(losses) / len(losses)

    model.train()
    return out


def set_lr(it, optimizer, cfg: DictConfig):
    """
    if cfg.decay_lr is None, do nothing
    """
    if cfg.get('decay_lr', None) is None:
        return
    
    lr = cfg.learning_rate
    warmup_iters = cfg.decay_lr.warmup_iters
    lr_decay_iters = cfg.decay_lr.lr_decay_iters
    min_lr = cfg.decay_lr.min_lr

    if it < warmup_iters:
        lr = lr * it / warmup_iters
    elif it > lr_decay_iters:
        lr = min_lr
    else:
        phase = math.pi * (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= phase <= math.pi
        coefficient = 0.5 * (1.0 + math.cos(phase))
        lr = min_lr + coefficient * (lr - min_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(cfg):
    train_set = get_dataset(cfg)
    sampler = RandomSampler(
        train_set, replacement=True, num_samples=cfg.max_iters * cfg.batch_size
    )
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, sampler=sampler)
    with torch.device(device):
        model = GPTV(cfg)
    
    optimizer = model.configure_optimizer(cfg)

    # logger.info(f"{torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB "
    #             "CUDA memory allocated for model and optimizer")
    # torch.cuda.reset_peak_memory_stats()

    store_losses = []
    model.train()
    pbar = tqdm(train_loader)
    for iter_num, batch in enumerate(pbar, start=1):
        set_lr(iter_num, optimizer, cfg)
        img = batch[0].to(device)
        x = batch[1].to(device)
        y = batch[2].to(device)
        logits, loss = model(img, x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter_num % cfg.eval_interval == 0:
            losses = estimate_loss(model, train_set, cfg)
            pbar.set_description(f"train loss {losses['train']:.4f}")
            store_losses.append(losses)

    # record final loss under larger evaluation samples
    final_losses = estimate_loss(model, train_set, cfg, eval_num_samples=cfg.final_eval_samples)
    logger.info(f"Final train loss {final_losses['train']}")

    # save checkpoint
    save_checkpoint(model, optimizer, iter_num, cfg)

    # save config yaml
    OmegaConf.save(cfg, os.path.join(cfg.out_dir, 'config.yaml'))

    # test_generation = generate_one_sequence(model, device, cfg)
    # with open(os.path.join(cfg.out_dir, 'test_generation.txt'), 'w') as f:
    #     f.write(test_generation)

    # logger.info(f"{torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB "
    #             "CUDA memory allocated for training")
    
    save_loss_fig(store_losses, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_log", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logger_level = "DEBUG"
    else:
        logger_level = "INFO"

    cfg = load_config(args.config)

    # torch.cuda.reset_peak_memory_stats() # reset memory counter
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    if args.name is not None:
        cfg.out_dir = os.path.join(cfg.out_dir, args.name)
    os.makedirs(cfg.out_dir, exist_ok=True)

    logger.remove()
    if args.save_log:
        logger.add(os.path.join(cfg.out_dir, 'log.txt'),
                   format="{level} - {message}", 
                   level=logger_level,
                   mode="w")
    logger.add(sys.stdout, format="{level} - {message}", level=logger_level)
    
    start = time.perf_counter()
    train(cfg)
    end = time.perf_counter()
    logger.info(f"Training time: {end - start:.2f} seconds")

'''
python train_GPTV.py --config configs/image_caption.yaml --name gptv --debug --save_log
'''
