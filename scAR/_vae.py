# -*- coding: utf-8 -*-

from scipy import stats
import torch.nn as nn
import torch
from torch.distributions import Normal, kl_divergence, Multinomial, Binomial, Poisson
from ._activation_functions import mytanh, hnormalization, mySoftplus

#########################################################################
## Variational antoencoder
#########################################################################

class VAE(nn.Module):

    def __init__(self, n_genes, fc1_dim, fc2_dim, enc_dim, scRNAseq_tech = 'scRNAseq', dropout_prob=0, model='binomial'):
        super().__init__()
        self.scRNAseq_tech = scRNAseq_tech
        self.model = model
        if fc1_dim==None and fc2_dim==None and enc_dim==None:
            fc1_dim, fc2_dim, enc_dim = 150, 100, 15
                
        self.encoder = Encoder(n_genes, fc1_dim, fc2_dim, enc_dim, dropout_prob)
        self.decoder = Decoder(n_genes, fc1_dim, fc2_dim, enc_dim, scRNAseq_tech, dropout_prob, model)
        
        print("..Running VAE using the following param set:")
        print("......scAR mode: ", scRNAseq_tech)
        print("......count model: ", model)
        print("......num_input_feature: ", n_genes)
        print("......NN_layer1: ", fc1_dim)
        print("......NN_layer2: ", fc2_dim)
        print("......latent_space: ", enc_dim)
        print("......dropout_prob: ", dropout_prob)

    def forward(self, x):
        z, mu, var = self.encoder(x)
        dec_nr, dec_prob, dec_dp = self.decoder(z)
        return z, dec_nr, dec_prob,  mu, var, dec_dp
    
    def inference(self, x, amb_prob, model='poisson'):
        
        # Estimate native signals
        z, dec_nr, dec_prob,  mu, var, dec_dp = self.forward(x)
        total_count_per_cell = x.sum(dim=1).view(-1, 1)
        prob_native = dec_prob*(1-dec_nr)
        expected_native_counts = (total_count_per_cell * prob_native).cpu().numpy()
        x_np = x.cpu().numpy()
        
        ### Calculate the Bayesian factors
        # The probability that observed UMI counts do not purely come from expected distribution of ambient signals.
        
        if model.lower() == 'binomial':
            amb_tot = torch.ceil(total_count_per_cell * dec_nr).cpu().numpy()
            error_term = ((x_np.sum()-amb_tot.sum())/len(x_np.flatten()))
            # H1: x is drawn from binomial distribution with prob > amb_prob  vs H2: x is drawn from binomial distribution with prob = amb_prob 
            probs_H1 = stats.binom.cdf(x_np-error_term, amb_tot, amb_prob.cpu().numpy())
            probs_H2 = stats.binom.pmf(x_np-error_term, amb_tot, amb_prob.cpu().numpy())

        elif model.lower() == 'poisson':
            expected_amb_counts = (total_count_per_cell * dec_nr * amb_prob).cpu().numpy()
            error_term = ((x_np.sum()-expected_amb_counts.sum())/len(x_np.flatten())) # adding this error term significantly improve the performance
            # H1: x is drawn from Poisson distribution with prob > amb_prob  vs H2: x is drawn from Poisson distribution with prob = amb_prob 
            probs_H1 = stats.poisson.cdf(x_np, expected_amb_counts + error_term)
            probs_H2 = stats.poisson.pmf(x_np, expected_amb_counts + error_term)

        bf = (probs_H1 + 1e-22)/(probs_H2 + 1e-22)
        
        
        return expected_native_counts, bf, dec_prob, dec_nr

#########################################################################
## Encoder
#########################################################################
class Encoder(nn.Module):
    """
    Encoder that takes the original expressions of feature barcodes and produces the encoding.

    Consists of 2 FC layers.
    """
    def __init__(self, n_genes, fc1_dim, fc2_dim, enc_dim, dropout_prob):
        super().__init__()
        self.activation = nn.SELU()
        self.fc1 = nn.Linear(n_genes, fc1_dim)
        self.bn1 = nn.BatchNorm1d(fc1_dim, momentum=0.01, eps=0.001)
        self.dp1 = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.bn2 = nn.BatchNorm1d(fc2_dim, momentum=0.01, eps=0.001)
        self.dp2 = nn.Dropout(p=dropout_prob)

        self.linear_means = nn.Linear(fc2_dim, enc_dim)
        self.linear_log_vars = nn.Linear(fc2_dim, enc_dim)
        self.z_transformation = nn.Softmax(dim=-1)
        
    def reparametrize(self, mu, log_vars):
        
        var = log_vars.exp() + 1e-4
        
        return torch.distributions.Normal(mu, var.sqrt()).rsample(), mu, var

    def forward(self, x):
        # encode
        x = (x+1).log2()   # log transformation of count data
        enc = self.fc1(x)
        enc = self.bn1(enc)
        enc = self.activation(enc)
        enc = self.dp1(enc)
        enc = self.fc2(enc)        
        enc = self.bn2(enc)        
        enc = self.activation(enc)
        enc = torch.clamp(enc, min=None, max=1e7)
        enc = self.dp2(enc)
        
        means = self.linear_means(enc)
        log_vars = self.linear_log_vars(enc)
        z, mu, var = self.reparametrize(means, log_vars)
        latent_transform = self.z_transformation(z)
        
        return latent_transform, mu, var


#########################################################################
## Decoder
#########################################################################

class Decoder(nn.Module):
    """
    A decoder model that takes the encodings and a batch (source) matrix and produces decodings.

    Made up of 2 FC layers.
    """
    def __init__(self, n_genes, fc1_dim, fc2_dim, enc_dim, scRNAseq_tech, dropout_prob, model):
        super().__init__()
        self.activation = nn.SELU()
        self.normalization_native_freq = hnormalization
        self.noise_activation = mytanh        
        self.activation_native_freq = nn.ReLU()        
        self.fc4 = nn.Linear(enc_dim, fc2_dim)
        self.bn4 = nn.BatchNorm1d(fc2_dim, momentum=0.01, eps=0.001)
        self.dp4 = nn.Dropout(p=dropout_prob)
        self.fc5 = nn.Linear(fc2_dim, fc1_dim)
        self.bn5 = nn.BatchNorm1d(fc1_dim, momentum=0.01, eps=0.001)
        self.dp5 = nn.Dropout(p=dropout_prob)
        self.noise_fc = nn.Linear(fc1_dim, 1)
        self.out_fc = nn.Linear(fc1_dim, n_genes)
        self.model = model
        if model.lower() == 'zeroinflatedpoisson':
            self.dropoutprob = nn.Linear(fc1_dim, 1)
            self.dropout_activation = nn.Sigmoid()
            

    def forward(self, z):
        # decode layers
        dec = self.fc4(z)
        dec = self.bn4(dec)
        dec = self.activation(dec)
        dec = self.fc5(dec)
        dec = self.bn5(dec)
        dec = self.activation(dec)
        dec = torch.clamp(dec, min=None, max=1e7)

        
        # final layers to produce prob parameters
        dec_prob = self.out_fc(dec)
        dec_prob = self.activation_native_freq(dec_prob)
        dec_prob = self.normalization_native_freq(dec_prob)
        
        # final layers to produce noise_ratio parameters
        dec_nr = self.noise_fc(dec)
        dec_nr = self.noise_activation(dec_nr)
        
        # final layers to learn the dropout probability
        if self.model.lower() == 'zeroinflatedpoisson':
            dec_dp = self.dropoutprob(dec)
            dec_dp = self.dropout_activation(dec_dp)
        else:
            dec_dp = None
            
        return dec_nr, dec_prob, dec_dp
