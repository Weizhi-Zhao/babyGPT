import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from models import GPT
from utils import resume_checkpoint
import argparse
from pathlib import Path
import torch
from torch import Tensor
from torch.nn import functional as F
import time
from typing import Optional, Generator
import pickle


TEMPERATURE = 1
TOP_K = 1
TEST_TIMES = 10

def sample(logits: Tensor) -> Tensor:
    """
    the logits must have 3 dim: batch, seq, vocab_size
    but the first two dim must be 1
    return a Tensor to keep device unchanged
    """
    assert logits.dim() == 3
    assert logits.size(0) == 1 and logits.size(1) == 1
    # sequeeze the seq dim
    probs = logits[:, 0, :] / TEMPERATURE
    if TOP_K is not None:
        v, _ = torch.topk(probs, min(TOP_K, probs.size(-1)))
        pivot = v[0, -1]
        probs[probs < pivot] = float('-inf')
    probs = F.softmax(probs, dim=-1)
    # sampling, so vocab dim becomes seq dim
    token_next = torch.multinomial(probs, num_samples=1)
    assert token_next.dim() == 2, "token_next must have batch and seq dim"
    assert token_next.size(0) == 1 and token_next.size(1) == 1
    return token_next

def prefill(model: GPT, x: Tensor, input_pos: Tensor) -> Tensor:
    # breakpoint()
    logits = model(x, input_pos=input_pos)
    return sample(logits)

def decode_one_token(model: GPT, token: Tensor, input_pos: Tensor) -> Tensor:
    assert input_pos.dim() == 1, "input_pos should be a 1D tensor"
    assert input_pos.size(0) == 1, "only decode one token"
    assert token.dim() == 2
    assert token.size(0) == 1 and token.size(1) == 1, "token should be a 2D tensor: [1, 1]"

    # breakpoint()
    logits = model(token, input_pos=input_pos)
    return sample(logits)

def decode_n_token(
    model: GPT,
    cur_token: Tensor,
    input_pos: Tensor,
    num_new_tokens: int,
    end_token: Optional[int] = None,
) -> Generator[Tensor, None, None]:
    for _ in range(num_new_tokens):
        next_token = decode_one_token(model, cur_token, input_pos)
        yield next_token.clone()
        if next_token == end_token:
            break
        input_pos += 1
        cur_token = next_token


@torch.no_grad
def generate(
    model: GPT, prompt: Tensor, max_tokens: int, end_token: Optional[int] = None
) -> Generator[int, None, None]:
    assert prompt.dim() == 1, "prompt should be a 1D tensor"
    device = prompt.device
    with torch.device(device):
        model.setup_caches(batch_size=1, block_size=model.cfg.block_size)

    T = prompt.size(0)
    T_NEW = max_tokens - T
    input_pos = torch.arange(0, T, device=device)

    next_token = prefill(model, prompt[None, :], input_pos)
    yield next_token

    input_pos = torch.tensor([T], dtype=torch.int, device=device)
    yield from decode_n_token(
        model,
        next_token,
        input_pos,
        T_NEW - 1,
    )


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # inference(ckpt=args.ckpt, device=device)

    checkpoint = torch.load(args.ckpt, mmap=True)
    with torch.device(device):
        model = GPT(checkpoint['config'])
    model.load_state_dict(checkpoint['model'], assign=True, strict=False)
    model = model.to(device)
    model.eval()
    with open(checkpoint['config'].meta_path, 'rb') as f:
        meta = pickle.load(f)
    ctoi = meta['ctoi']
    itoc = meta['itoc']
    encode = lambda s: [ctoi[c] for c in s]
    decode = lambda t: "".join(itoc[i] for i in t)

    start = time.perf_counter()
    result = ''
    for _ in range(TEST_TIMES):
        # for t in generate(model, torch.tensor(encode('A'), device=device), max_tokens=512):
        #     print(decode([t.item()]), end='', flush=True)

        # result = ''.join((decode([t.item()]) for t in generate(model, torch.tensor(encode('A'), device=device), max_tokens=512)))
        

        generator = stream_generator(
            model,
            device,
            "A",
            encode,
            decode,
            max_new_tokens=512-1,
            top_k=TOP_K,
            temperature=TEMPERATURE
        )
        result = ''.join(generator)
    print(f"TIME: {time.perf_counter() - start:.2f}s")
    print(result)


"""
python generate.py --ckpt output/shakespeare_char/kv_cache/ckpt.pt
"""
