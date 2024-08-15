from models.GPT import RoPE
from omegaconf import OmegaConf
import torch

if __name__ == '__main__':
    cfg = OmegaConf.create("block_size: 2\nn_head: 2\nn_embd: 4\nrope_base: 10000")
    x = torch.tensor([[[[1, 2], [3, 4]], [[1, 2], [3, 4]]]], dtype=torch.float32)
    rope = RoPE(cfg)
    print(rope.freqs_cis)
    print(rope(x))


"""
[x1, x2]^T * [y1, y2] = x1*y1 + x2*y2
(x1 + ix2) * (y1 + iy2)*
= (x1 + ix2) * (y1 - iy2)
= x1*y1 + x2*y2


theta = 10000^(-2i/4) = 1
e^(t*theta) = 
 0   ,  0
 0.54,  0.84
-0.41,  0.90
-0.98,  0.14
-0.65, -0.75

 0   ,  0
 0.99,  0.01
 0.99,  0.02
 0.99,  0.03
"""