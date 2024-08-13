# import tiktoken
# enc = tiktoken.encoding_for_model("gpt-4o")
# print(enc.max_token_value)
# enc = tiktoken.encoding_for_model("gpt-4")
# print(enc.max_token_value)
# enc = tiktoken.encoding_for_model("gpt-2")
# print(enc.max_token_value)
import pickle

meta_path = 'data/chinese_poetry/meta.pkl'
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
    itoc = meta['itoc']
    print(itoc[-1])