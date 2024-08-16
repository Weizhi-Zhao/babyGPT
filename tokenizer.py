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
        # assert meta.get('special_tokens') is not None, "special_tokens must be in meta.pkl"
        # self.special_tokens: dict = meta['special_tokens']

        # thanks to llama3, add Chinese and <s> </s>
        pat_str = r"<s>|</s>|[\p{Han}]|[\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|" \
              + r"\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|" \
              + r"\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|" \
              + r"\u2026|\u2014|\uff5e|\ufe4f|\uffe5]" \
              + r"|(?i:'s|'t|'re|'ve|'m|'ll|'d)" \
              + r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1}" \
              + r"| ?[^\s\p{L}\p{N}\p{Han}</s><s>]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        self.pattern = regex.compile(pat_str)

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

if __name__ == '__main__':
    tokenizer = Tokenizer(Path('data/image_caption/meta.pkl'))
    print(tokenizer.encode('\n'))
    print(tokenizer.decode([1]))