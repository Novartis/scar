import torch

def mytanh(x):
    a = x.exp()
    tanh = (a - 1/a) / (a + 1/a)
    tanh = (1 + tanh) / 2
    return tanh

def hnormalization(x):
#     x = torch.nan_to_num(x, nan=1e-7)
    return x/(x.sum(dim=1).view(-1,1) + 1e-5)

def mySoftplus(x):
    mask0 = x<=-5 #
    x[mask0] = 0
    x[~mask0] = (x[~mask0].exp()+1).log()
    return x