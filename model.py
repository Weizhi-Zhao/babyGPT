from omegaconf import OmegaConf, DictConfig
from torch.nn import functional as F
import torch.nn as nn
import torch
from loguru import logger
import inspect


"""
nn.MultiheadAttention's loss is slightly higher than myAttention
Does nn.MultiheadAttention inference faster? yes
how to use Nested Tensor. due to Nested Tensor conflict with attn mask, put it later
what if use nn.TransformerDecoder and nn.TransformerDecoderLayer? No it contains cross attention
is DROP out inplace better? no, inplace is slower and worse
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
        self.register_buffer("mask", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))

        # cause nn.MultiheadAttention only dropout attention weights
        self.dropout = nn.Dropout(cfg.dropout) 

    def forward(self, x: torch.Tensor):
        x, _ = self.c_attn(x, x, x, need_weights=False,
                           attn_mask=self.mask, is_causal=True)
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
        # what if no view here? view doesn't mater
        """
        masked_fill: The shape of mask must be broadcastable with the shape of the underlying tensor.
        In short, if a PyTorch operation supports broadcast, then its Tensor arguments can be
        automatically expanded to be of equal sizes (without making copies of the data).
        so the view should be unnecessary
        """
        # why attn mask need grad? or it doesn't? it really doesn't require grad
        self.register_buffer("mask", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
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
        attn_weights = attn_weights.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        y = attn_weights @ v
        # what if no contiguous? must use contiguous()
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.act = nn.GELU()
        # self.act = nn.ReLU()
        self.fc2 = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        # nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

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
        # self.attn = MyCausalSelfAttention(cfg)
        self.attn = CausalSelfAttention(cfg)
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
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd, device='meta'),
            wpe = nn.Embedding(cfg.block_size, cfg.n_embd),
            drop = nn.Dropout(cfg.dropout), # drop token+position
            blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln = nn.LayerNorm(cfg.n_embd, bias=cfg.bias), # final layer norm
        ))

        # meta device really saves memory
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # really better? weight tying, YES!!!
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        logger.info(f"GPT parameter number: {self.get_num_params()/1e6:.2f}M")

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.lm_head.weight)

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
        pos = torch.arange(0, T, dtype=torch.int64, device=device)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(tokens)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.ln(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            """
            input: (B, T, vocab_size) -> (B * T, vocab_size)
            target: (B, T) -> (B * T)
            ignore_index: some sentence does not fill the length of block_size
            so maybe they are padded with -1
            """
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            return logits
    
    def configure_optimizer(self, cfg):
        weight_decay = cfg.weight_decay
        learning_rate = cfg.learning_rate
        betas = (cfg.beta1, cfg.beta2)
        # start with all parameters
        param_dict: dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that shouldn't be trained
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        logger.debug(
            f"parameters don't need grad: "
            f"{[pn for pn, p in self.named_parameters() if not p.requires_grad]}")
        
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        # it is a little worse
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        # optim_groups = [
        #     {"params": decay_params, "weight_decay": weight_decay},
        #     {"params": nodecay_params, "weight_decay": weight_decay},
        # ]


        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        logger.info(
            f"num decayed parameter tensors: "
            f"{len(decay_params)}, with {num_decay_params:,} parameters")
        logger.info(
            f"num non-decayed parameter tensors: "
            f"{len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        """
        fused: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
        If the user specifies True for both foreach and fused, 
        we will prioritize fused over foreach, as it is typically faster.
        HOWEVER, since the fused implementation is relatively new, we want to 
        give it sufficient bake-in time, so we default to foreach and NOT fused 
        when the user has not specified either flag.
        """
        use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        # it seems that fused supports cpu
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
    
    @torch.no_grad
    def generate(self, tokens, max_new_tokens, temperature=1.0, topk=None):
        """
        tokens: (B, T), dtype=torch.int64
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # crop conditioning tokens to block_size
            if tokens.size(1) > self.cfg.block_size:
                tokens_cond = tokens[:, -self.cfg.block_size:]
            else:
                tokens_cond = tokens

            logits = self(tokens_cond)
            logits = logits[:, -1, :] / temperature
            if topk is not None:
                v, _ = torch.topk(logits, k=min(topk, logits.size(-1)))
                # torch.topk retuens are sorted
                # so -1 is the smallest
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            # although num_samples=1, it returns a (B, 1) tensor
            next_token = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens
