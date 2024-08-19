from itertools import islice
import regex
import json
from tqdm import tqdm

"""
https://opendatalab.com/OpenDataLab/WanJuan1_dot_0/explore/main
"""

NUM_JSON = 25000

def main():
    data = ""
    pattern = r"[\x00-\xff]"
    with open("part-000003-28feace1.jsonl", 'r', encoding='utf-8') as f:
        for line in tqdm(islice(f, NUM_JSON), total=NUM_JSON):
            json_dict = json.loads(line)
            data += regex.sub(pattern=pattern, repl="", string=json_dict['content'])
            # data += json_dict['content']
            # print(data)
    with open("data.txt", 'w', encoding='utf-8') as f:
        f.write(data)

if __name__ == "__main__":
    main()
    