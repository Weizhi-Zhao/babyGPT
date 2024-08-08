import torch
from torch.nn import functional as F

a = torch.arange(-3, 3).repeat(10, 1)
a = a.float()
b = torch.ones(10, 1)
a[a < b] = float('-inf')
prob = F.softmax(a, dim=1)
next_t = torch.multinomial(prob, 1)
print(next_t)
# print(torch.topk(param, k=5))
