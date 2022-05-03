# -*- coding: utf-8 -*-
"""
The variational autoencoder
"""

import numpy as np
from scipy import stats

import torch
from torch import nn

from ._activation_functions import mytanh, hnormalization, MySoftplus

#########################################################################
## Variational autoencoder
#########################################################################


class VAE(nn.Module):
    """A class of variational autoencoder

    Args:
        n_features (int): number of features (e.g. mRNA, sgRNA, ADT, tag, ...)
        nn_layer1 (int): number of neurons in the 1st layer. Default: 150
        nn_layer2 (int): number of neurons in the 2nd layer. Default: 100
        latent_dim (int): number of neurons in the bottleneck layer. Default: 15
        feature_type (str): the feature to be denoised, \
            either of 'mRNA', 'sgRNA', 'ADT', 'tag'. Default: 'mRNA'
        count_model (str): the model to generate the UMI count, \
            either of "binomial", "poisson", "zeroinflatedpoisson". Default: 'binomial'
        verbose (bool): whether to print the details of neural networks

    Returns:
        Object: a class object
    """

    def __init__(
        self,
        n_features,
        nn_layer1=150,
        nn_layer2=100,
        latent_dim=15,
        dropout_prob=0,
        feature_type="mRNA",
        count_model="binomial",
        sparsity=0.5,
        verbose=True,
    ):

        super().__init__()
        assert feature_type.lower() in [
            "mrna",
            "mrnas",
            "sgrna",
            "sgrnas",
            "adt",
            "adts",
            "tag",
            "tags",
        ]
        assert count_model.lower() in ["binomial", "poisson", "zeroinflatedpoisson"]
        # self.feature_type = feature_type
        # self.count_model = count_model
        self.encoder = Encoder(
            n_features, nn_layer1, nn_layer2, latent_dim, dropout_prob
        )
        self.decoder = Decoder(
            n_features,
            nn_layer1,
            nn_layer2,
            latent_dim,
            dropout_prob,
            feature_type,
            count_model,
            sparsity,
        )

        if verbose:
            print("..Running VAE using the following param set:")
            print("......denoised count type: ", feature_type)
            print("......count model: ", count_model)
            print("......num_input_feature: ", n_features)
            print("......NN_layer1: ", nn_layer1)
            print("......NN_layer2: ", nn_layer2)
            print("......latent_space: ", latent_dim)
            print("......dropout_prob: ", dropout_prob)

    def forward(self, input_matrix):
        """forward function"""
        sampling, means, var = self.encoder(input_matrix)
        dec_nr, dec_prob, dec_dp = self.decoder(sampling)
        return dec_nr, dec_prob, means, var, dec_dp

    @torch.no_grad()
    def inference(
        self, input_matrix, amb_prob, count_model_inf="poisson", adjust="micro"
    ):
        """
        Inference of presence of native signals
        """
        assert count_model_inf.lower() in ["poisson", "binomial", "zeroinflatedpoisson"]
        assert adjust in [False, "global", "micro"]

        # Estimate native signals
        dec_nr, dec_prob, _, _, _ = self.forward(input_matrix)

        # Copy tensor to CPU
        input_matrix_np = input_matrix.cpu().numpy()
        noise_ratio = dec_nr.cpu().numpy().reshape(-1, 1)
        nat_prob = dec_prob.cpu().numpy()
        amb_prob = amb_prob.cpu().numpy().reshape(1, -1)

        total_count_per_cell = input_matrix_np.sum(axis=1).reshape(-1, 1)
        expected_native_counts = total_count_per_cell * (1 - noise_ratio) * nat_prob
        expected_amb_counts = total_count_per_cell * noise_ratio * amb_prob
        tot_amb = expected_amb_counts.sum(axis=1).reshape(-1, 1)

        if not adjust:
            adjust = 0
        elif adjust == "global":
            adjust = (total_count_per_cell.sum() - tot_amb.sum()) / len(
                input_matrix_np.flatten()
            )
        elif adjust == "micro":
            adjust = (total_count_per_cell - tot_amb) / input_matrix_np.shape[1]
            adjust = np.repeat(adjust, input_matrix_np.shape[1], axis=1)

        ### Calculate the Bayesian factors
        # The probability that observed UMI counts do not purely
        # come from expected distribution of ambient signals.
        # H1: x is drawn from distribution (binomial or poission or
        # zeroinflatedpoisson)with prob > amb_prob
        # H2: x is drawn from distribution (binomial or poission or
        # zeroinflatedpoisson) with prob = amb_prob

        if count_model_inf.lower() == "binomial":
            probs_h1 = stats.binom.logcdf(input_matrix_np, tot_amb + adjust, amb_prob)
            probs_h2 = stats.binom.logpmf(input_matrix_np, tot_amb + adjust, amb_prob)

        elif count_model_inf.lower() == "poisson":
            probs_h1 = stats.poisson.logcdf(
                input_matrix_np, expected_amb_counts + adjust
            )
            probs_h2 = stats.poisson.logpmf(
                input_matrix_np, expected_amb_counts + adjust
            )

        elif count_model_inf.lower() == "zeroinflatedpoisson":
            raise NotImplementedError

        bayesian_factor = np.clip(probs_h1 - probs_h2 + 1e-22, -709.78, 709.78)
        bayesian_factor = np.exp(bayesian_factor)

        return (
            expected_native_counts,
            bayesian_factor,
            dec_prob.cpu().numpy(),
            dec_nr.cpu().numpy(),
        )


#########################################################################
## Encoder
#########################################################################
class Encoder(nn.Module):
    """
    Encoder that takes the original expressions of feature barcodes and produces the encoding.

    Consists of 2 FC layers.
    """

    def __init__(self, n_features, nn_layer1, nn_layer2, latent_dim, dropout_prob):
        """initialization"""
        super().__init__()
        self.activation = nn.SELU()
        self.fc1 = nn.Linear(n_features, nn_layer1)
        self.bn1 = nn.BatchNorm1d(nn_layer1, momentum=0.01, eps=0.001)
        self.dp1 = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(nn_layer1, nn_layer2)
        self.bn2 = nn.BatchNorm1d(nn_layer2, momentum=0.01, eps=0.001)
        self.dp2 = nn.Dropout(p=dropout_prob)

        self.linear_means = nn.Linear(nn_layer2, latent_dim)
        self.linear_log_vars = nn.Linear(nn_layer2, latent_dim)
        self.z_transformation = nn.Softmax(dim=-1)

    def reparametrize(self, means, log_vars):
        """reparameterization"""
        var = log_vars.exp() + 1e-4
        return torch.distributions.Normal(means, var.sqrt()).rsample(), var

    def forward(self, input_matrix):
        """forward function"""
        input_matrix = (input_matrix + 1).log2()  # log transformation of count data
        enc = self.fc1(input_matrix)
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
        sampling, var = self.reparametrize(means, log_vars)
        latent_transform = self.z_transformation(sampling)

        return latent_transform, means, var


#########################################################################
## Decoder
#########################################################################
class Decoder(nn.Module):
    """
    A decoder model that takes the encodings and a batch (source) matrix and produces decodings.

    Made up of 2 FC layers.
    """

    def __init__(
        self,
        n_features,
        nn_layer1,
        nn_layer2,
        latent_dim,
        dropout_prob,
        feature_type,
        count_model,
        sparsity,
    ):
        """initialization"""
        super().__init__()
        self.activation = nn.SELU()
        self.normalization_native_freq = hnormalization()
        self.noise_activation = mytanh()
        if feature_type.lower() in ["sgrna", "sgrnas", "tag", "tags"]:
            self.activation_native_freq = nn.ReLU()
        elif feature_type.lower() in ["mrna", "mrnas", "adt", "adts"]:
            self.activation_native_freq = MySoftplus(sparsity)
        self.fc4 = nn.Linear(latent_dim, nn_layer2)
        self.bn4 = nn.BatchNorm1d(nn_layer2, momentum=0.01, eps=0.001)
        self.dp4 = nn.Dropout(p=dropout_prob)
        self.fc5 = nn.Linear(nn_layer2, nn_layer1)
        self.bn5 = nn.BatchNorm1d(nn_layer1, momentum=0.01, eps=0.001)
        self.dp5 = nn.Dropout(p=dropout_prob)

        self.noise_fc = nn.Linear(nn_layer1, 1)
        self.out_fc = nn.Linear(nn_layer1, n_features)
        self.count_model = count_model
        if count_model.lower() == "zeroinflatedpoisson":
            self.dropoutprob = nn.Linear(nn_layer1, 1)
            self.dropout_activation = mytanh()

    def forward(self, sampling):
        """forward function"""
        # decoder
        dec = self.fc4(sampling)
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
        if self.count_model.lower() == "zeroinflatedpoisson":
            dec_dp = self.dropoutprob(dec)
            dec_dp = self.dropout_activation(dec_dp)
            dec_dp = torch.nan_to_num(dec_dp, nan=1e-7)
        else:
            dec_dp = None

        return dec_nr, dec_prob, dec_dp
