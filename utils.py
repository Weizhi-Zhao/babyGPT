from loguru import logger
from matplotlib import pyplot as plt
from omegaconf import OmegaConf, DictConfig, ListConfig
from pathlib import Path
from typing import Literal
import os
import pickle
import torch
from models import GPT

def recursive_load_config(cfg):
    if isinstance(cfg, DictConfig):
        temp_cfg = OmegaConf.create()
        for k, v in cfg.items():
            if isinstance(v, str) and Path(v).suffix == '.yaml':
                if not Path(v).exists():
                    logger.warning(f"Config file {v} does not exist")
                    continue
                sub_cfg = OmegaConf.load(v)
                sub_cfg = recursive_load_config(sub_cfg)
                temp_cfg = OmegaConf.merge(temp_cfg, sub_cfg)
            elif isinstance(v, (DictConfig, ListConfig)):
                cfg[k] = recursive_load_config(v)
        cfg = OmegaConf.merge(cfg, temp_cfg)
    elif isinstance(cfg, ListConfig):
        pass
        # temp_cfg = cfg.copy()
        # for i, item in enumerate(cfg):
        #     if isinstance(item, str) and Path(item).exists() and Path(item).suffix == '.yaml':
        #         sub_cfg = OmegaConf.load(item)
        #         sub_cfg = recursive_load_config(sub_cfg)
        #         temp_cfg = OmegaConf.merge(temp_cfg, sub_cfg)
        #     elif isinstance(item, (DictConfig, ListConfig)):
        #         cfg[i] = recursive_load_config(item)
    else:
        raise ValueError(f"Unsupported type: {type(cfg)}, "
                         "only DictConfig and ListConfig are supported")
    return cfg


def load_config(config_path):
    cfg = OmegaConf.load(config_path)
    cfg = recursive_load_config(cfg)
    return cfg

def get_vocab_size(path: Path | str):
    if isinstance(path, str):
        path = Path(path)

    if path.suffix == '.yaml':
        path = load_config(path).meta_path
    elif path.suffix == '.pkl':
        pass

    with open(path, 'rb') as f:
        meta = pickle.load(f)

    return meta['vocab_size']

def generate_one_sequence(model, device, cfg: DictConfig) -> str:
    model.eval()
    if cfg.get("meta_path", None) is not None:
        with open(cfg.meta_path, 'rb') as f:
            meta = pickle.load(f)
        ctoi = meta['ctoi']
        itoc = meta['itoc']
        encode = lambda s: [ctoi[c] for c in s]
        decode = lambda l: ''.join([itoc[i] for i in l])
    
    cond = encode(cfg.condition_prompt)
    cond = torch.tensor(cond, dtype=torch.long, device=device)[None, ...]
    y = model.generate(cond, cfg.max_new_tokens, cfg.temperature, cfg.top_k)
    return decode(y[0].tolist())


def save_loss_fig(losses: list[dict[Literal["train", "test"]: float]], cfg: DictConfig):
    fig, ax = plt.subplots()

    ax.plot([l['train'] for l in losses], 'b', label='Train Loss')
    if 'test' in losses[0]:
        ax.plot([l['test'] for l in losses], 'r', label='Test Loss')
    if 'val' in losses[0]:
        ax.plot([l['val'] for l in losses], 'r', label='val Loss')

    ax.set_xlabel(
        f"{cfg.eval_num_samples} samples every {cfg.eval_interval} iterations")
    ax.set_ylabel('Loss')
    ax.set_title('Train and Test Loss')
    ax.legend()
    fig.savefig(os.path.join(cfg.out_dir, 'loss.png'),
                dpi=300, bbox_inches='tight')


def tensor_mem(x: torch.Tensor):
    return x.element_size() * x.nelement()


def save_checkpoint(model, optimizer, iter_num, cfg: DictConfig):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter_num": iter_num,
        "config": cfg
    }
    torch.save(checkpoint, os.path.join(cfg.out_dir, 'ckpt.pt'))
    logger.info(f"save checkpoint to {os.path.join(cfg.out_dir, 'ckpt.pt')}")


def resume_checkpoint(ckpt_path: Path, device, inference=False):
    # let's just map_location = device like karpathy
    checkpoint = torch.load(ckpt_path, map_location=device,
                            weights_only=False, mmap=True)
    cfg = checkpoint["config"]
    with torch.device("meta"):
        model = GPT(cfg)
    model.load_state_dict(checkpoint["model"], assign=True)
    model = model.to(device=device)

    if inference:
        with open(cfg.meta_path, 'rb') as f:
            meta = pickle.load(f)
        return model, meta
    
    optimizer = model.configure_optimizer(cfg)
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer, checkpoint["iter_num"]


if __name__ == '__main__':
    # Example usage
    # cfg = load_config('configs/shakespeare_char.yaml')
    # print(OmegaConf.to_yaml(cfg))
    print(get_vocab_size('configs/shakespeare_char.yaml'))