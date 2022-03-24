"""Activation functions"""
# -*- coding: utf-8 -*-


def mytanh(x):
    """mytanh"""
    a = x.exp()
    tanh = (a - 1 / a) / (a + 1 / a)
    tanh = (1 + tanh) / 2
    return tanh


def hnormalization(x):
    """hnormalization"""
    return x / (x.sum(dim=1).view(-1, 1) + 1e-5)


def mySoftplus(x):
    """mySoftplus"""
    mask0 = x <= -5
    x[mask0] = 0
    x[~mask0] = (x[~mask0].exp() + 1).log()
    return x
