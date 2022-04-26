# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats
import torch.nn as nn
import torch
from ._activation_functions import mytanh, hnormalization

#########################################################################
## Variational autoencoder
#########################################################################

class VAE(nn.Module):
    """Variational autoencoder"""
    def __init__(
        self,
        n_genes,
        fc1_dim,
        fc2_dim,
        enc_dim,
        scRNAseq_tech="scRNAseq",
        dropout_prob=0,
        model="binomial",
    ):
        """initialization"""
        super().__init__()
        assert scRNAseq_tech.lower() in ["scrnaseq", "cropseq", "citeseq"]
        assert model.lower() in ["binomial", "poisson", "zeroinflatedpoisson"]
        self.scRNAseq_tech = scRNAseq_tech
        self.model = model
        if fc1_dim == None and fc2_dim == None and enc_dim == None:
            fc1_dim, fc2_dim, enc_dim = 150, 100, 15

        self.encoder = Encoder(n_genes, fc1_dim, fc2_dim, enc_dim, dropout_prob)
        self.decoder = Decoder(
            n_genes, fc1_dim, fc2_dim, enc_dim, scRNAseq_tech, dropout_prob, model
        )

        print("..Running VAE using the following param set:")
        print("......scAR mode: ", scRNAseq_tech)
        print("......count model: ", model)
        print("......num_input_feature: ", n_genes)
        print("......NN_layer1: ", fc1_dim)
        print("......NN_layer2: ", fc2_dim)
        print("......latent_space: ", enc_dim)
        print("......dropout_prob: ", dropout_prob)

    def forward(self, x):
        """forward function"""
        z, mu, var = self.encoder(x)
        dec_nr, dec_prob, dec_dp = self.decoder(z)
        return z, dec_nr, dec_prob, mu, var, dec_dp

    @torch.no_grad()
    def inference(self, x, amb_prob, model="poisson", adjust="micro"):
        """
        Inference of presence of native signals
        """
        assert model.lower() in ["poisson", "binomial", "zeroinflatedpoisson"]
        assert adjust in [False, "global", "micro"]

        # Estimate native signals
        _, dec_nr, dec_prob, _, _, _ = self.forward(x)

        # Copy tensor to CPU
        x_np = x.cpu().numpy()
        nr = dec_nr.cpu().numpy().reshape(-1, 1)
        nat_prob = dec_prob.cpu().numpy()
        amb_prob = amb_prob.cpu().numpy().reshape(1, -1)

        total_count_per_cell = x_np.sum(axis=1).reshape(-1, 1)
        expected_native_counts = total_count_per_cell * (1 - nr) * nat_prob
        expected_amb_counts = total_count_per_cell * nr * amb_prob
        tot_amb = expected_amb_counts.sum(axis=1).reshape(-1, 1)

        if not adjust:
            adjust = 0
        elif adjust == "global":
            adjust = (total_count_per_cell.sum() - tot_amb.sum()) / len(x_np.flatten())
        elif adjust == "micro":
            adjust = (total_count_per_cell - tot_amb) / x_np.shape[1]
            adjust = np.repeat(adjust, x_np.shape[1], axis=1)

        ### Calculate the Bayesian factors
        # The probability that observed UMI counts do not purely 
        # come from expected distribution of ambient signals.
        # H1: x is drawn from distribution (binomial or poission or
        # zeroinflatedpoisson)with prob > amb_prob
        # H2: x is drawn from distribution (binomial or poission or
        # zeroinflatedpoisson) with prob = amb_prob

        if model.lower() == "binomial":
            probs_H1 = stats.binom.logcdf(x_np, tot_amb + adjust, amb_prob)
            probs_H2 = stats.binom.logpmf(x_np, tot_amb + adjust, amb_prob)

        elif model.lower() == "poisson":
            probs_H1 = stats.poisson.logcdf(x_np, expected_amb_counts + adjust)
            probs_H2 = stats.poisson.logpmf(x_np, expected_amb_counts + adjust)

        elif model.lower() == "zeroinflatedpoisson":
            raise NotImplementedError

        bf = np.clip(probs_H1 - probs_H2 + 1e-22, -709.78, 709.78)
        bf = np.exp(bf)

        return expected_native_counts, bf, dec_prob.cpu().numpy(), dec_nr.cpu().numpy()


#########################################################################
## Encoder
#########################################################################
class Encoder(nn.Module):
    """
    Encoder that takes the original expressions of feature barcodes and produces the encoding.

    Consists of 2 FC layers.
    """

    def __init__(self, n_genes, fc1_dim, fc2_dim, enc_dim, dropout_prob):
        """initialization"""
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
        """reparameterization"""
        var = log_vars.exp() + 1e-4
        return torch.distributions.Normal(mu, var.sqrt()).rsample(), mu, var

    def forward(self, x):
        """forward function"""
        x = (x + 1).log2()  # log transformation of count data
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

    def __init__(
        self, n_genes, fc1_dim, fc2_dim, enc_dim, scRNAseq_tech, dropout_prob, model
    ):
        """initialization"""
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
        if model.lower() == "zeroinflatedpoisson":
            self.dropoutprob = nn.Linear(fc1_dim, 1)
            self.dropout_activation = mytanh

    def forward(self, z):
        """forward function"""
        # decoder
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
        if self.model.lower() == "zeroinflatedpoisson":
            dec_dp = self.dropoutprob(dec)
            dec_dp = self.dropout_activation(dec_dp)
            dec_dp = torch.nan_to_num(dec_dp, nan=1e-7)
        else:
            dec_dp = None

        return dec_nr, dec_prob, dec_dp
