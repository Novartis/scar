# -*- coding: utf-8 -*-
"""
Customized activation functions
"""

import torch

def mytanh(var_in):
    """customized tanh activation, output range: (0, 1)"""
    var_tanh = torch.tanh(var_in)
    var_out = (1 + var_tanh) / 2
    return var_out


def hnormalization(var_in):
    """Summation and normalization"""
    return var_in / (var_in.sum(dim=1).view(-1, 1) + 1e-5)


def mysoftplus(var_in):
    """customized softplus activation, output range: [0, inf)"""
    var_sp = torch.nn.functional.softplus(var_in)
    threshold = torch.nn.functional.softplus(torch.tensor(-5.0))
    zero = torch.zeros_like(threshold)
    var_out = torch.where(var_sp <= threshold, zero, var_sp)
    return var_out
    