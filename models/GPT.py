from omegaconf import OmegaConf, DictConfig
from torch.nn import functional as F
import torch.nn as nn
import torch
from loguru import logger
import inspect


class RoPE(nn.Module):
    # thanks to llama3: https://github.com/meta-llama/llama3/blob/main/llama/model.py
    def __init__(self, cfg):
        super().__init__()
        self.block_size = cfg.block_size
        assert (
            cfg.n_embd % cfg.n_head == 0
        ), "Embedding size must be an integer multiple of the number of heads"
        self.head_dim = cfg.n_embd // cfg.n_head
        rope_base = cfg.rope_base

        assert self.head_dim % 2 == 0, "head_dim must be an even number"
        thetas = 1.0 / (
            rope_base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        t = torch.arange(self.block_size).float()
        # precompute m*theta for token dim and position dim
        freqs = torch.outer(t, thetas) # [block_size, n_embd//2], val = pos*theta
        # cis: complex number, freq_cis = e^(i*freqs)
        freqs_cis = torch.polar(abs=torch.ones_like(freqs), angle=freqs)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x: torch.Tensor):
        assert x.size(3) == self.head_dim, "x.size(3) must be equal to head_dim"
        assert x.size(2) <= self.block_size, "x.size(2) must be less than or equal to block_size"
        freqs_cis = self.freqs_cis[:x.size(2), :] # [seq_size, head_dim]
        x = x.reshape(*x.shape[:-1], -1, 2)
        x = torch.view_as_complex(x)
        x = x * freqs_cis
        x = torch.view_as_real(x).flatten(3)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, \
            "Embedding size must be an integer multiple of the number of heads"

        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

        self.drop_p = cfg.dropout
        self.drop = nn.Dropout(cfg.dropout)
        self.n_head = cfg.n_head

    def forward(self, x: torch.Tensor, rope: RoPE):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        # transpose B and n_head together, for batched multi-head matrix multiplication
        # (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = rope(q)
        k = rope(k) 

        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.drop_p if self.training else 0, is_causal=True
        )

        y = y.transpose(1, 2).view(B, T, C)
        # y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

        self._reset_parameters()

    def _reset_parameters(self):
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
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x, rope: RoPE):
        x = x + self.attn(self.ln1(x), rope)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.vocab_size is not None
        assert cfg.block_size is not None
        self.cfg = cfg

        self.rope = RoPE(cfg)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd, device='meta'),
            drop = nn.Dropout(cfg.dropout), # drop token+position
            blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln = nn.LayerNorm(cfg.n_embd, bias=cfg.bias), # final layer norm
        ))

        # meta device really saves memory
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        logger.info(f"GPT parameter number: {self.get_num_params()/1e6:.2f}M")

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.lm_head.weight)

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    def forward(self, tokens, targets=None):
        B, T = tokens.size()
        assert T <= self.cfg.block_size, \
            f"Cannot forward sequence of length {T}, block size is only {self.cfg.block_size}"

        # forward the GPT model itself
        tok_emb = self.transformer.wte(tokens)
        # tested, useful
        x = self.transformer.drop(tok_emb)
        for block in self.transformer.blocks:
            x = block(x, self.rope)
        x = self.transformer.ln(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
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