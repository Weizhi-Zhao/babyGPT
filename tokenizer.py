"""
for simple dataset like shakepeare, use itoa and ctoi in meta.pkl as tokenizer
"""

from pathlib import Path
import pickle
import regex

class Tokenizer:
    """
    handle special tokens: <s> </s>
    """
    def __init__(self, meta_path: Path):
        with open(meta_path, 'rb') as f:
            meta: dict = pickle.load(f)
        self.ctoi = meta['ctoi']
        self.itoc = meta['itoc']
        # thanks to llama3, add Chinese and <s> </s>
        pat_str = meta['pat_str']
        self.pattern = regex.compile(pat_str)

        self._vocab_size = meta['vocab_size']

        self._start_token = self.ctoi['<s>']
        self._end_token = self.ctoi['</s>']

    def encode(self, text: str) -> list:
        text_list = self.pattern.findall(text)
        tokens = [self.ctoi[c] for c in text_list]
        return tokens

    def decode(self, tokens: list[int]) -> str:
        text = ''.join([self.itoc[i] for i in tokens])
        return text
    
    @property
    def start_token(self) -> int:
        return self._start_token

    @property
    def end_token(self) -> int:
        return self._end_token
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size

if __name__ == '__main__':
    tokenizer = Tokenizer(Path('data/image_caption/meta.pkl'))
    print(tokenizer.encode('\n'))
    print(tokenizer.decode([1]))