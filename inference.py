import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from models import GPT
from utils import resume_checkpoint
import argparse
from pathlib import Path
import torch
from torch.nn import functional as F
import time


def stream_generator(
    model: GPT,
    device,
    prompt: str,
    encode: callable,
    decode: callable,
    max_new_tokens: int,
    top_k: int,
    temperature=1.0,
    start: str | None = None,
    end: str | None = None,
):
    if start is not None:
        prompt = start + prompt
    
    tokens = encode(prompt)
    tokens = torch.tensor(tokens, device=device)
    for _ in range(max_new_tokens):
        if tokens.size(0) > model.cfg.block_size:
            tokens = tokens[-model.cfg.block_size:]
        logits = model(tokens[None, :])
        logits = logits[0, -1, :] / temperature
        if top_k is not None:
            min_v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
            logits[logits < min_v[-1]] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat((tokens, next_token), dim=0)
        
        next_char = decode(next_token.tolist())
        if end is not None and end == next_char:
            return
        yield next_char


def inference(ckpt: Path, device):
    model, meta = resume_checkpoint(ckpt, device, inference=True)
    model.eval()
    ctoi = meta['ctoi']
    itoc = meta['itoc']
    encode = lambda s: [ctoi[c] for c in s]
    decode = lambda t: "".join(itoc[i] for i in t)
    while True:
        prompt = input("请输入提示词：")
        if prompt == "exit":
            break
        generator = stream_generator(
            model,
            device,
            prompt + '\n',
            encode,
            decode,
            max_new_tokens=500,
            top_k=50,
            temperature=0.7,
            start="s",
            end="e",
        )

        print('\n' + prompt + '\n', end="", flush=True)
        for char in generator:
            print(char, end="", flush=True)
            # time.sleep(0.1)
        print('\n\n')
        # time.sleep(3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inference(ckpt=args.ckpt, device=device)


"""
python inference.py --ckpt output/chinese_poetry/Chinese_poetry/ckpt.pt
"""