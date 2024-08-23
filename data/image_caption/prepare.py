import yaml
import json
import os
import regex
import pickle
import sys
sys.path.append('../..')
from tokenizer import Tokenizer
import torch


def main():
    # thanks to llama3, add Chinese and <s> </s>
    pat_str = (
        r"<s>|</s>|[\p{Han}]|[\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|"
        + r"\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|"
        + r"\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|"
        + r"\u2026|\u2014|\uff5e|\ufe4f|\uffe5]"
        + r"|(?i:'s|'t|'re|'ve|'m|'ll|'d)"
        + r"|[^\r\n\p{L}\p{N}]?[a-zA-Z]+|\p{N}{1}"
        + r"| ?[^\s\p{L}\p{N}\p{Han}</s><s>]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    )
    pattern = regex.compile(pat_str)

    # NLP dataset
    with open("nlp_train.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    prepared_data = []
    unique_chars = set()
    for k, v in data.items():
        img_path = os.path.join("Train", k)
        caption: str = v["请简要描述这张图片"]
        # clean the data
        caption = caption.replace("<s>", "")
        caption = caption.replace("</s>", " ")
        caption = caption.strip()

        caption = "<s>" + caption + "</s>"
        prepared_data.append({"img": img_path, "caption": caption})
        caption_split_list = pattern.findall(caption)
        unique_chars.update(set(caption_split_list))
    with open("train_data.json", "w", encoding="utf-8") as f:
        json.dump(prepared_data, f, ensure_ascii=False, indent=4)

    # val_data
    validation_data = []
    with open("nlp_val.yaml", "r", encoding="utf-8") as f:
        val_data = yaml.safe_load(f)
    for k, v in val_data.items():
        img_path = os.path.join("Val", k)
        caption: str = v["请简要描述这张图片"]
        caption = caption.replace("<s>", "")
        caption = caption.replace("</s>", " ")
        caption = caption.strip()

        caption = "<s>" + caption + "</s>"
        validation_data.append({"img": img_path, "caption": caption})
    with open("val_data.json", "w", encoding="utf-8") as f:
        json.dump(validation_data, f, ensure_ascii=False, indent=4)

    # pretrain data
    with open("./pretrain/data.txt", "r", encoding="utf-8") as f:
        pretrain_data = f.read()
    unique_chars.update(set(pattern.findall(pretrain_data)))

    # build tokenizer
    unique_chars = sorted(list(unique_chars))
    print(f"Unique Chars: {unique_chars}")
    vocab_size = len(unique_chars)
    print(f"Vocab Size: {vocab_size}")

    ctoi = {c: i for i, c in enumerate(unique_chars)}
    itoc = {i: c for i, c in enumerate(unique_chars)}

    # save meta information
    meta = {
        "vocab_size": vocab_size,
        "ctoi": ctoi,
        "itoc": itoc,
        "pattern_str": pat_str,
    }
    with open("meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    # test
    tn = Tokenizer("meta.pkl")
    print(pattern.findall("</s>从一些测试中得出结论？! </s>"))
    print(tn.decode(tn.encode("</s>从一些测试中得出结论？</s>")))

    # save pretrain data
    pretrain_data_tensor = torch.tensor(tn.encode(pretrain_data), dtype=torch.long)
    torch.save(pretrain_data_tensor, "pretrain_data.pt")


if __name__ == "__main__":
    main()
