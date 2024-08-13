from loguru import logger
from pathlib import Path
import argparse
import numpy as np
import pickle
import requests
import sys
import torch
import tiktoken

def main():
    # download data
    data_path = Path(__file__).parent / 'data.txt'
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    if data_path.exists() == False:
        with open(data_path, 'w', encoding='utf-8') as f:
            f.write(requests.get(data_url).text)

    # read data
    with open(data_path, 'r', encoding='utf-8') as f:
        data = f.read()
    logger.info(f"length of data in character: {len(data):,}") # ;, is insert thousand separators

    tokenizer = tiktoken.encoding_for_model("gpt-2")

    # get all unique chars in dataset
    logger.info(f"vocab size: {tokenizer.max_token_value}")

    # create mapping between characters and integers
    logger.debug(f"test encoder & decoder: "
                 f"{tokenizer.decode(tokenizer.encode('Hello World!'))}")

    # create train and test splits
    n = len(data)
    train_data = data[:round(0.9 * n)]
    test_data = data[round(0.9 * n):]

    # encode both to integers
    train_tokens = tokenizer.encode(train_data)
    test_tokens = tokenizer.encode(test_data)
    logger.info(f"train set has {len(train_tokens):,} tokens")
    logger.info(f"test set has {len(test_tokens):,} tokens")

    # export to .pt files
    train_tokens = torch.tensor(train_tokens, dtype=torch.long)
    test_tokens = torch.tensor(test_tokens, dtype=torch.long)
    torch.save(train_tokens, Path(__file__).parent / 'train.pt')
    torch.save(test_tokens, Path(__file__).parent / 'test.pt')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logger_level = "DEBUG" if args.debug else "INFO"
    logger.remove()
    logger.add(sys.stdout, format="{level} - {message}", level=logger_level)
    main()