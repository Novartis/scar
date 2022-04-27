# -*- coding: utf-8 -*-
"""Loss functions"""

import torch
from torch.distributions import Normal, kl_divergence, Binomial, Poisson
from pyro.distributions.zero_inflated import ZeroInflatedPoisson

def kld(means, var):
    """KL divergence"""
    mean = torch.zeros_like(means)
    scale = torch.ones_like(var)
    return kl_divergence(Normal(means, torch.sqrt(var)), Normal(mean, scale)).sum(dim=1)


def get_reconstruction_loss(
    input_matrix, dec_nr, dec_prob, amb_prob, dec_dp, count_model
):
    """reconstruction loss"""
    tot_count = input_matrix.sum(dim=1).view(-1, 1)
    prob_tot = dec_prob * (1 - dec_nr) + amb_prob * dec_nr

    if count_model.lower() == "zeroinflatedpoisson":
        recon_loss = -ZeroInflatedPoisson(
            rate=tot_count * prob_tot / (1 - dec_dp), gate=dec_dp, validate_args=False
        ).log_prob(input_matrix)
        recon_loss = torch.nan_to_num(recon_loss, nan=1e-7, posinf=1e15, neginf=-1e15)
        recon_loss = recon_loss.sum(axis=1).mean()

    elif count_model.lower() == "binomial":
        recon_loss = -Binomial(tot_count, probs=prob_tot, validate_args=False).log_prob(
            input_matrix
        )
        recon_loss = torch.nan_to_num(recon_loss, nan=1e-7, posinf=1e15, neginf=-1e15)
        recon_loss = recon_loss.sum(axis=1).mean()

    elif count_model.lower() == "poisson":
        recon_loss = -Poisson(rate=tot_count * prob_tot, validate_args=False).log_prob(
            input_matrix
        )  # add 1 to avoid a situation where all counts are zeros
        recon_loss = torch.nan_to_num(recon_loss, nan=1e-7, posinf=1e15, neginf=-1e15)
        recon_loss = recon_loss.sum(axis=1).mean()

    return recon_loss


def loss_fn(
    input_matrix,
    dec_nr,
    dec_prob,
    means,
    var,
    amb_prob,
    reconstruction_weight,
    kld_weight=1e-5,
    dec_dp=None,
    count_model="binomial",
):
    """loss function"""

    recon_loss = get_reconstruction_loss(
        input_matrix, dec_nr, dec_prob, amb_prob, dec_dp=dec_dp, count_model=count_model
    )
    kld_loss = kld(means, var).sum()
    total_loss = recon_loss * reconstruction_weight + kld_loss * kld_weight

    return recon_loss, kld_loss, total_loss
