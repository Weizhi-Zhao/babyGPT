
# is warmup lr really works? yes
# is lr decay really works?
# TODO: what is ctx?
# TODO: what is scaler?
# TODO: what and why grad_clip?
# TODO: add resume
# add tiktoken: not good for Chinese and shakespeare
# TODO: add chinese dataset
# TODO: KV cache
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from datasets import DATASETS
from loguru import logger
from models import GPT
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from utils import load_config, generate_one_sequence, save_loss_fig, save_checkpoint
import argparse
import math
import sys
import time
import torch
from omegaconf import OmegaConf, DictConfig


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataset(cfg):
    dataset_cls = DATASETS[cfg.dataset_name]
    train_set = dataset_cls(cfg.train_set_path, cfg.block_size)
    test_set = dataset_cls(cfg.test_set_path, cfg.block_size)
    return train_set, test_set


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad
def estimate_loss(model, train_set, test_set, cfg, eval_num_samples=None):
    model.eval()
    out = {}

    if eval_num_samples is None:
        eval_num_samples = cfg.eval_num_samples * cfg.batch_size

    eval_train_sampler = RandomSampler(
        train_set, replacement=False, num_samples=eval_num_samples)
    eval_train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, sampler=eval_train_sampler)
    eval_test_sampler = RandomSampler(
        test_set, replacement=False, num_samples=eval_num_samples
    )
    eval_test_loader = DataLoader(
        test_set, batch_size=cfg.batch_size, sampler=eval_test_sampler
    )

    losses = []
    for x, y in eval_train_loader:
        x = x.to(device)
        y = y.to(device)
        logits, loss = model(x, targets=y)
        losses.append(loss.item())
    out['train'] = sum(losses) / len(losses)

    losses = []
    for x, y in eval_test_loader:
        x = x.to(device)
        y = y.to(device)
        logits, loss = model(x, targets=y)
        losses.append(loss.item())
    out['test'] = sum(losses) / len(losses)

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
    train_set, test_set = get_dataset(cfg)
    train_sampler = RandomSampler(
        train_set, replacement=True, num_samples=cfg.max_iters * cfg.batch_size)
    # pin memory? useless
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, sampler=train_sampler, num_workers=cfg.num_workers)
    with torch.device(device):
        model = GPT(cfg)
    optimizer = model.configure_optimizer(cfg)
    
    logger.info(f"{torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB "
                "CUDA memory allocated for model and optimizer")
    torch.cuda.reset_peak_memory_stats()

    store_losses = []
    model.train()
    pbar = tqdm(train_loader)
    for iter_num, batch in enumerate(pbar):
        set_lr(iter_num, optimizer, cfg)
        x = batch[0].to(device)
        y = batch[1].to(device)
        logits, loss = model(x, targets=y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter_num % cfg.eval_interval == 0:
            losses = estimate_loss(model, train_set, test_set, cfg)
            pbar.set_description(f"train loss {losses['train']:.4f}, test loss {losses['test']:.4f}")
            store_losses.append(losses)

    # record final loss under larger evaluation samples
    final_losses = estimate_loss(model, train_set, test_set, cfg, eval_num_samples=cfg.final_eval_samples)
    logger.info(f"Final train loss {final_losses['train']}, test loss {final_losses['test']}")

    # save checkpoint
    save_checkpoint(model, optimizer, iter_num, cfg)

    save_loss_fig(store_losses, cfg)

    # save config yaml
    OmegaConf.save(cfg, os.path.join(cfg.out_dir, 'config.yaml'))
    
    test_generation = generate_one_sequence(model, device, cfg)
    with open(os.path.join(cfg.out_dir, 'test_generation.txt'), 'w') as f:
        f.write(test_generation)

    logger.info(f"{torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB "
                "CUDA memory allocated for training")


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

    torch.cuda.reset_peak_memory_stats() # reset memory counter
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
python train.py --config configs/shakespeare_char.yaml --name test --debug --save_log

1. bias, relu, drop
python train.py --config configs/shakespeare_char.yaml --name initial --gpu 0 --debug --save_log
python train.py --config configs/sc_bias.yaml --name bias --gpu 2 --debug --save_log
python train.py --config configs/shakespeare_char.yaml --name relu --gpu 3 --debug --save_log
python train.py --config configs/sc_drop0.yaml --name drop0 --gpu 1 --debug --save_log

2. my attention vs nn.MultiheadAttention
python train.py --config configs/shakespeare_char.yaml --name nn_attn --gpu 0 --debug --save_log

3. inplace dropout
python train.py --config configs/shakespeare_char.yaml --name inplace_drop --gpu 0 --debug --save_log

4. no weight tying
python train.py --config configs/shakespeare_char.yaml --name no_weight_tyign --gpu 0 --debug --save_log

5. weight decay 1e-1 -> 1e-2
python train.py --config configs/sc_decay1e-2.yaml --name decay1e-2 --gpu 1 --debug --save_log

6. custom weight init
python train.py --config configs/shakespeare_char.yaml --name cus_wei_init --gpu 0 --debug --save_log

7. all biases and layernorms don't decay is better?
python train.py --config configs/shakespeare_char.yaml --name no_separate_decay --gpu 0 --debug --save_log

8. use official nn.TransformerDecoder and nn.TransformerDecoderLayer
python train.py --config configs/shakespeare_char.yaml --name official_transformer --gpu 0 --debug --save_log

9. Does lr decay really work?
python train.py --config configs/shakespeare_char.yaml --name decay_lr --gpu 0 --debug --save_log

10. Does lr warmup really work?
python train.py --config configs/shakespeare_char.yaml --name lr_warmup --gpu 0 --debug --save_log

11. Can I save memory from weight tying?
python train.py --config configs/shakespeare_char.yaml --name save_wt_mem --gpu 0 --debug --save_log

12. will customize initialization help?
python train.py --config configs/shakespeare_char.yaml --name cust_init --gpu 0 --debug --save_log

13. test np.memmap memory usage
python train.py --config configs/shakespeare_char.yaml --name memmap --debug --save_log
python train.py --config configs/sc_torch_load.yaml --name memmap --gpu 0 --debug --save_log

14. gpt-4 tiktoken
python train.py --config configs/shakespeare.yaml --name tiktoken --debug --save_log

15. Chinese poetry dataset
python train.py --config configs/chinese_poetry.yaml --name Chinese_poetry --debug --save_log
python train.py --config configs/chinese_poetry.yaml --name cp_rope --debug --save_log

16. RoPE
python train.py --config configs/sc_rope.yaml --name RoPE --debug --save_log
python train.py --config configs/sc_rope.yaml --name RoPE_rm1drop --debug --save_log
python train.py --config configs/sc_rope.yaml --name RoPE_drop03 --debug --save_log
python train.py --config configs/sc_rope.yaml --name RoPE_drop04 --debug --save_log

17. test dropout
python train.py --config configs/sc_drop.yaml --name drop03 --debug --save_log

18. test silu
python train.py --config configs/shakespeare_char.yaml --name silu --debug --save_log
'''