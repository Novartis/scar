import torch.nn as nn
import torch
from torch.distributions import Normal, kl_divergence, Multinomial, Binomial, Poisson

#########################################################################
### Loss functions
#########################################################################

def kld(mu, var):
    
    mean = torch.zeros_like(mu)
    scale = torch.ones_like(var)
    return kl_divergence(Normal(mu, torch.sqrt(var)), Normal(mean, scale)).sum(dim=1)

def get_reconstruction_loss(x, dec_nr, dec_prob, amb_prob, model='binomial'):
    ### TODO: consider zero-inflated models
    
    tot_count = x.sum(dim=1).view(-1,1)
    prob_tot = dec_prob*(1-dec_nr)+ amb_prob*dec_nr
    
    if model.lower() == 'binomial':
        recon_loss = -Binomial(tot_count, probs=prob_tot, validate_args=False).log_prob(x)
        recon_loss = recon_loss.sum(axis=1).mean()
    elif model.lower() == 'poisson':
        recon_loss = -Poisson(rate=tot_count*prob_tot, validate_args=False).log_prob(x)  # add 1 to avoid a situation where all counts are zeros
        recon_loss = torch.nan_to_num(recon_loss, nan=1e-7, posinf=1e15, neginf=-1e15)
        recon_loss = recon_loss.sum(axis=1).mean()
    elif model.lower() == 'multinomial':
        recon_loss = 0
        for tot, prob, x_0 in zip(tot_count, prob_tot, x):
            recon_loss += - Multinomial(probs=prob, validate_args=False).log_prob(x_0)/len(tot_count)
            
    return recon_loss
    
def loss_fn(x, dec_nr, dec_prob, mu, var, amb_prob, reconstruction_weight, kld_weight=1e-5):
    
    recon_loss = get_reconstruction_loss(x, dec_nr, dec_prob, amb_prob)
    kld_loss = kld(mu, var).sum()
    
    total_loss = recon_loss*reconstruction_weight + kld_loss*kld_weight
    
    return recon_loss, kld_loss, total_loss    