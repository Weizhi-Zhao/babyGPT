from itertools import islice
import regex
import json
from tqdm import tqdm

"""
https://opendatalab.com/OpenDataLab/WanJuan1_dot_0/explore/main
"""

NUM_JSON = 20000
# NUM_JSON = 2

def main():
    data = ""
    pattern = r"[^\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b\u4e00-\u9fa5]"
    with open("part-000001-28feace1.jsonl", 'r', encoding='utf-8') as f:
        for line in tqdm(islice(f, NUM_JSON), total=NUM_JSON):
            json_dict = json.loads(line)
            data += regex.sub(pattern=pattern, repl="", string=json_dict['content'])
            # data += json_dict['content']
            # print(data)
    with open("data.txt", 'w', encoding='utf-8') as f:
        f.write(data)

if __name__ == "__main__":
    main()
    