import json
from itertools import islice

with open("./data/part-000003-28feace1.jsonl", "r", encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        breakpoint()