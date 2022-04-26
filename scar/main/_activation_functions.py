# -*- coding: utf-8 -*-
"""
Customized activation functions
"""

import torch
from torch.nn.functional import tanh, softplus

def mytanh(var_in):
    """customized tanh activation, output range: (0, 1)"""
    var_tanh = tanh(var_in)
    var_out = (1 + var_tanh) / 2
    return var_out


def hnormalization(var_in):
    """Summation and normalization"""
    return var_in / (var_in.sum(dim=1).view(-1, 1) + 1e-5)


def mySoftplus(var_in):
    """customized softplus activation, output range: [0, inf)"""
    var_sp = softplus(var_in)
    threshold = softplus(torch.tensor(-5.0))
    zero = torch.zeros_like(threshold)
    var_out = torch.where(var_sp <= threshold, zero, var_sp)
    return var_out
    