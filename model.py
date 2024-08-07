import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import OmegaConf, DictConfig


"""
TODO: confirm that MultiheadAttention and myAttention are equivalent
TODO: test MultiheadAttention performance with different configurations
TODO: is disable bias really better?
TODO: relu ot gelu?
TODO: how to use Nested Tensor
TODO: what if use nn.TransformerDecoder and nn.TransformerDecoderLayer?
"""


class CausalSelfAttention(nn.Module):
    # masked self attention
    def __init__(self, cfg: DictConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, \
            "Embedding size must be an integer multiple of the number of heads"

        self.c_attn = nn.MultiheadAttention(cfg.n_embd,
                                            cfg.n_head,
                                            dropout=cfg.dropout,
                                            bias=cfg.bias,
                                            batch_first=True)
        self.dropout = nn.Dropout(cfg.dropout) # cause nn.MultiheadAttention only dropout attention weights

    def forward(self, x: torch.Tensor):
        x = self.c_attn(x, x, x, need_weights=False, is_causal=True)
        x = self.dropout(x)
        return x


class MyCausalSelfAttention(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, \
            "Embedding size must be an integer multiple of the number of heads"
        
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

        self.attn_drop = nn.Dropout(cfg.dropout)
        self.proj_drop = nn.Dropout(cfg.dropout)

        self.n_embd = cfg.n_embd
        self.n_head = cfg.n_head

        """
        reasons for register buffer instead of self.mask = torch.tril(torch.ones(...
        1. buffer will be save and load with model
        2. buffer will handle device and parallel
        3. buffer will appear in state_dict
        """
        # TODO: what if no view here?
        """
        masked_fill: The shape of mask must be broadcastable with the shape of the underlying tensor.
        In short, if a PyTorch operation supports broadcast, then its Tensor arguments can be
        automatically expanded to be of equal sizes (without making copies of the data).
        so the view should be unnecessary
        """
        self.register_buffer("mask", torch.tril(torch.ones(
            cfg.block_size, cfg.block_size)).view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
        torch.split
        torch.tensor_split
        torch.chunk
        """
        what is th difference between torch.split, torch.tensor_split and torch.chunk?
        split and tensor_split split tensor by size
        chunk split tensor by number
        list of numbers can be provided into split and tensor_split
        in split, numbers are exact sizes
        for tensor_split, numbers are indices
        there are also some other differences when dim is not divisible by split_size

        nn.Linear only transforms the last dimension
        """
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        # transpose B and n_head together, for batched multi-head matrix multiplication
        # (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        attn_weights = q @ k.transpose(-2, -1) / (k.size(-1) ** 0.5)
        # block_size for max sequence length, T <= block_size
        attn_weights = attn_weights.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        y = attn_weights @ v
        # TODO: what if no contiguous?
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj_drop(self.c_proj(y))

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        match cfg.act:
            case "gelu":
                self.act = nn.GELU()
            case "relu":
                self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = MyCausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.vocab_size is not None
        assert cfg.block_size is not None
        self.cfg = cfg

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe = nn.Embedding(cfg.block_size, cfg.n_embd),
            drop = nn.Dropout(cfg.dropout), # drop token+position
            blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln = nn.LayerNorm(cfg.n_embd, bias=cfg.bias), # final layer norm
        ))

        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # TODO: really better?
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        """
        TODO: is this code better?
        # init all weights
        self.apply(self._init_weights)
        TODO: is this code better?
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        """

        logger.info(f"GPT parameter number: {self.get_num_params():.2f}M")

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    def forward(self, tokens, targets=None):
        device = tokens.device
        B, T = tokens.size()
        assert T <= self.cfg.block_size, \
            f"Cannot forward sequence of length {T}, block size is only {self.cfg.block_size}"
