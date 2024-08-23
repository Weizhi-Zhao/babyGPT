import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from models import GPTV
from utils import resume_checkpoint
import argparse
from pathlib import Path
import torch
from torch.nn import functional as F
import time
from PIL import Image
from torchvision import transforms
import yaml
from tokenizer import Tokenizer
from tqdm import tqdm


def stream_generator(
    model: GPTV,
    device,
    prompt: str,
    image: torch.Tensor,
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
        logits = model(image, tokens[None, :])
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


def inference(model, img: Path, encode, decode, device):
    model.eval()
    img = Image.open(img).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = transform(img).unsqueeze(0).to(device)
    prompt = ""
    generator = stream_generator(
        model,
        device,
        prompt,
        img,
        encode,
        decode,
        max_new_tokens=256,
        top_k=20,
        temperature=0.5,
        start="<s>",
        end="</s>",
    )
    out = ""
    for char in generator:
        out += char
    
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--img_path", type=str, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.ckpt, map_location=device,
                            weights_only=False, mmap=True)
    
    cfg = checkpoint["config"]
    cfg.pretrain = False
    with torch.device("meta"):
        model = GPTV(cfg)
    model.load_state_dict(checkpoint["model"], assign=True)
    model = model.to(device=device)

    tokenizer = Tokenizer(cfg.meta_path)
    model.eval()
    encode = tokenizer.encode
    decode = tokenizer.decode
    result = {}
    img_list = [p for p in Path(args.img_path).glob('*.jpg')]
    for img in tqdm(img_list):
        res = inference(model, img, encode, decode, device=device)
        result[img.name] = res
    with open("train_result.yaml", "w", encoding='utf-8') as f:
        yaml.dump(result, f, allow_unicode=True)

"""
python infer_GPTV.py --ckpt checkpoints/ckpt_gptv_8_20.pt --img data/image_caption/Val
python infer_GPTV.py --ckpt checkpoints/ckpt_gptv_8_20.pt --img data/image_caption/Train
"""
