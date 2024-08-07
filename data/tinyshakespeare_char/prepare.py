
import requests
import argparse
import pickle
import logging
logger = logging.getLogger(__name__)

from pathlib import Path
import numpy as np

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

    # get all unique chars in dataset
    unique_chars = sorted(list(set(data)))
    vocab_size = len(unique_chars)
    logger.info(f"all the unique characters: {''.join(unique_chars)}")
    logger.info(f"vocab size: {vocab_size}")

    # create mapping between characters and integers
    ctoi = {c: i for i, c in enumerate(unique_chars)}
    itoc = {i: c for i, c in enumerate(unique_chars)}
    encode = lambda s: [ctoi[c] for c in s] # s for string
    decode = lambda l: "".join(itoc[i] for i in l) # l for list
    logger.debug(f"test encoder & decoder: {decode(encode('Hello World!'))}")

    # create train and test splits
    n = len(data)
    train_data = data[:round(0.9 * n)]
    test_data = data[round(0.9 * n):]

    # encode both to integers
    train_tokens = encode(train_data)
    test_tokens = encode(test_data)
    logger.info(f"train set has {len(train_tokens):,} tokens")
    logger.info(f"test set has {len(test_tokens):,} tokens")

    # export to bin(npy) files
    train_tokens = np.array(train_tokens, dtype=np.uint16)
    test_tokens = np.array(test_tokens, dtype=np.uint16)
    train_tokens.tofile(Path(__file__).parent / 'train.npy')
    test_tokens.tofile(Path(__file__).parent / 'test.npy')

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
    logger_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=logger_level, format="%(levelname)s - %(name)s -- %(message)s")
    main()