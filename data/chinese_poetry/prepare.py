from loguru import logger
from pathlib import Path
import argparse
import numpy as np
import pickle
import os
import sys
import torch
import json

# ! s for <s>, e for </s>

def main():
    # read data
    data_path = Path(__file__).parent
    data = ""
    unique_chars = set()
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if not file.endswith('.json'):
                continue
            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                poetrys = json.load(f)
            for p in poetrys:
                x = p['title'] + '\n' + p['content']
                unique_chars.update(set(x))
                data += 's' + x + 'e'
    logger.info(f"length of data in character: {len(data):,}") # ;, is insert thousand separators

    # get all unique chars in dataset
    unique_chars = sorted(list(unique_chars))
    vocab_size = len(unique_chars)

    # create mapping between characters and integers
    ctoi = {c: i for i, c in enumerate(unique_chars)}
    # special tokens
    ctoi['s'] = vocab_size
    ctoi['e'] = vocab_size + 1
    itoc = {i: c for i, c in enumerate(unique_chars)}
    itoc[vocab_size] = 's'
    itoc[vocab_size + 1] = 'e'
    encode = lambda s: [ctoi[c] for c in s] # s for string
    decode = lambda l: "".join(itoc[i] for i in l) # l for list
    logger.debug(f"test encoder & decoder: {decode(encode('白日依山尽，黄河入海流。'))}")

    vocab_size += 2 # +2 for 's' and 'e'
    logger.info(f"vocab size: {vocab_size}")

    # create train and test splits
    n = len(data)
    train_data = data[:round(0.9 * n)]
    test_data = data[round(0.9 * n):]

    # encode both to integers
    train_tokens = encode(train_data)
    test_tokens = encode(test_data)
    logger.info(f"train set has {len(train_tokens):,} tokens")
    logger.info(f"test set has {len(test_tokens):,} tokens")

    # export to .pt files
    torch.save(torch.tensor(train_tokens, dtype=torch.long), Path(__file__).parent / 'train.pt')
    torch.save(torch.tensor(test_tokens, dtype=torch.long), Path(__file__).parent / 'test.pt')

    # save meta information
    meta = {
        "vocab_size": vocab_size,
        "ctoi": ctoi,
        "itoc": itoc
    }
    with open(Path(__file__).parent / 'meta.pkl', 'wb') as f:
        pickle.dump(meta, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logger_level = "DEBUG" if args.debug else "INFO"
    logger.remove()
    logger.add(sys.stdout, format="{level} - {message}", level=logger_level)
    main()