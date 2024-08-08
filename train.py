
# TODO: is warmup lr really works?
# TODO: is lr decay really works?
# TODO: what is ctx?
# TODO: what is scaler?
# TODO: what and why grad_clip?

import torch
from utils import load_config
from dataset import DATASETS
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from model import GPT
import argparse
import logging
from tqdm import tqdm
logger = logging.getLogger(__name__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_loader(cfg):
    dataset_cls = DATASETS[cfg.dataset_name]
    train_set = dataset_cls(cfg.train_set_path, cfg.block_size)
    val_set = dataset_cls(cfg.val_set_path, cfg.block_size)
    # TODO: pin memory
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, val_loader


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad
def estimate_loss(model, train_loader, val_loader):
    out = {}
    model.eval()
    
    losses = []
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        logits, loss = model(x, y)
        losses.append(loss.item())
    out['train'] = sum(losses) / len(losses)

    losses = []
    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)
        logits, loss = model(x, y)
        losses.append(loss.item())
    out['val'] = sum(losses) / len(losses)

    model.train()
    return out


def train(cfg):
    train_loader, val_loader = get_loader(cfg)
    # TODO: does this saves memory?
    with torch.device(device):
        model = GPT(cfg)
    optimizer = model.configure_optimizer(cfg)

    for epoch in range(1, cfg.epochs):
        pbar = tqdm(train_loader)
        for iter_num, batch in enumerate(pbar):
            x = batch[0].to(device)
            y = batch[1].to(device)
            logits, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Epoch {epoch} iter {iter_num} loss {loss.item()}")
    
        losses = estimate_loss(model, train_loader, val_loader)
        logger.info(f"Epoch {epoch} train loss {losses['train']}, val loss {losses['val']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.debug:
        logger_level = logging.DEBUG
    else:
        logger_level = logging.INFO
    logging.basicConfig(level=logger_level, format="%(levelname)s - %(name)s -- %(message)s")
    cfg = load_config(args.config)

    train(cfg)