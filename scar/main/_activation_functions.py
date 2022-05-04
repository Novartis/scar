# -*- coding: utf-8 -*-
"""
Customized activation functions
"""

import torch


class mytanh(torch.nn.Module):
    def __init__(self):
        super().__init__()  # init the base class

    def forward(self, input_x):
        var_tanh = torch.tanh(input_x)
        output = (1 + var_tanh) / 2
        return output


class hnormalization(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_x):
        return input_x / (input_x.sum(dim=1).view(-1, 1) + 1e-5)


class mysoftplus(torch.nn.Module):
    def __init__(self, sparsity=0.9):
        super().__init__()  # init the base class
        self.sparsity = sparsity

    def forward(self, input_x):
        return self._mysoftplus(input_x)

    def _mysoftplus(self, input_x):
        """customized softplus activation, output range: [0, inf)"""
        var_sp = torch.nn.functional.softplus(input_x)
        threshold = torch.nn.functional.softplus(
            torch.tensor(-(1 - self.sparsity) * 10.0, device=input_x.device)
        )
        var_sp = var_sp - threshold
        zero = torch.zeros_like(threshold)
        var_out = torch.where(var_sp <= zero, zero, var_sp)
        return var_out
