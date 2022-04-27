# -*- coding: utf-8 -*-

import os, sys, time
import torch
import contextlib
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Union
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tqdm.contrib import DummyTqdmFile
from ._vae import VAE
from ._loss_functions import loss_fn

# Writing progressbar into stdout rather than stderr, from https://github.com/tqdm/tqdm/blob/master/examples/redirect_print.py
@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err


# scar class
class model:

    """
    scar class, Single cell Ambient Remover [Sheng2022].

    Parameters
    ----------
    raw_count
        Raw count matrix (nd.array, pd.DataFrame, or a path).
    empty_profile
        Empty profile (a vector or matrix, nd.array, pd.DataFrame, or a path). Default: None, averaging raw counts to estimate the ambinet profile.
    NN_layer1
        Neuron number of the first hidden layer (int). Default: 150.
    NN_layer2
        Neuron number of the second hidden layer (int). Default: 100.
    latent_space
        Neuron number of the latent space (int). Default: 15.
    scRNAseq_tech
        One of the following:
            'scRNAseq' -- any mRNA counts, including mRNA counts in single cell CRISPR screens and CITE-seq experiments. Default.
            'CITEseq' -- protein counts for CITEseq
            'CROPseq' -- sgRNA/identity barcode counts, or any data types of super high sparsity. E.g., in cell indexing experiments, we would expect a single true signal (1) and many negative signals (0) for each cell,
    model
        Count model, one of the following:
            'binomial' -- binomial model. Defualt.
            'poisson' -- poisson model
            'zeroinflatedpoisson' -- zeroinflatedpoisson model, choose this one when the raw counts are sparse

    Examples
    --------
    >>> from scar import model
    >>> scarObj = model(adata.X.to_df(), empty_profile)
    >>> scarObj.train()
    >>> scarObj.inference()
    >>> adata.layers["X_scar_denoised"] = scarObj.native_counts
    >>> adata.obsm["X_scar_assignment"] = scarObj.feature_assignment  # in 'cropseq' mode

    """

    def __init__(
        self,
        raw_count: Union[str, np.ndarray, pd.DataFrame],
        empty_profile: Optional[Union[str, np.ndarray, pd.DataFrame]] = None,
        NN_layer1: int = 150,
        NN_layer2: int = 100,
        latent_space: int = 15,
        scRNAseq_tech: str = "scRNAseq",
        model: str = "binomial",
        cellID: Optional[Union[str, list]] = None,
        feature_names: str = None,
        num_input_feature: int = 100,
        device: str = None,
        n_batch_train: int = None,
        n_batch_val: int = None,
        batch_size: int = None,
        runtime: int = None,
        trained_model = None,
        native_counts = None,
        bayesfactor = None,
        native_frequencies = None,
        noise_ratio = None,
        feature_assignment = None
    ):
        """initialize object"""

        if isinstance(raw_count, str):
            raw_count = pd.read_pickle(raw_count)
        elif isinstance(raw_count, np.ndarray):
            raw_count = pd.DataFrame(
                raw_count,
                index=range(raw_count.shape[0]),
                columns=range(raw_count.shape[1]),
            )
        elif isinstance(raw_count, pd.DataFrame):
            pass
        else:
            raise TypeError(
                "Expecting str or np.array or pd.DataFrame object, but get a {}".format(
                    type(raw_count)
                )
            )
        raw_count = raw_count.fillna(0)  # missing vals -> zeros

        if isinstance(empty_profile, str):
            empty_profile = pd.read_pickle(empty_profile)
            empty_profile = empty_profile.fillna(0).values  # missing vals -> zeros
        elif isinstance(empty_profile, pd.DataFrame):
            empty_profile = empty_profile.fillna(0).values  # missing vals -> zeros
        elif isinstance(empty_profile, np.ndarray):
            empty_profile = np.nan_to_num(empty_profile)  # missing vals -> zeros
        elif not empty_profile:
            print(" ... Evaluate empty profile from cells")
            empty_profile = raw_count.sum() / raw_count.sum().sum()
            empty_profile = empty_profile.fillna(0).values
        else:
            raise TypeError(
                "Expecting str / np.array / None / pd.DataFrame, but get a {}".format(
                    type(empty_profile)
                )
            )

        if empty_profile.squeeze().ndim == 1:
            empty_profile = (
                empty_profile.squeeze()
                .reshape(1, -1)
                .repeat(raw_count.shape[0], axis=0)
            )

        self.cellID = list(raw_count.index)
        self.feature_names = list(raw_count.columns)
        self.num_input_feature = raw_count.shape[1]
        self.NN_layer1 = NN_layer1
        self.NN_layer2 = NN_layer2
        self.latent_space = latent_space
        self.scRNAseq_tech = scRNAseq_tech
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Loading numpy to tensor on GPU
        self.raw_count = torch.from_numpy(raw_count.values).int().to(self.device)
        self.empty_profile = torch.from_numpy(empty_profile).float().to(self.device)

    def train(
        self,
        batch_size: int = 64,
        train_size: float = 0.998,
        shuffle: bool = True,
        kld_weight: float = 1e-5,
        lr: float = 1e-3,
        lr_step_size: int = 5,
        lr_gamma: float = 0.97,
        epochs: int = 800,
        reconstruction_weight: float = 1,
        dropout_prob: float = 0,
        TensorBoard: bool = False,
        save_model: bool = False
    ):

        """
        Training scar model

        Parameters
        ----------
        batch_size
            Batch_size (int). Default: 64.
        train_size
            Proportion of train samples (float). Default: 0.998.
        shuffle
            Whether shuffle the data (bool). Default: True.
        kld_weight
            The weight on KL-divergence (float, positive). Default: 1e-5.
        lr
            Initial learning rate (float). Default: 1e-3.
        lr_step_size
            Period of learning rate decay (float). Default: 5. See https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html.
        lr_gamma
            Multiplicative factor of learning rate decay (float). Default: 0.97. See https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html.
        epochs
            Training iterations (float). Default: 800.
        reconstruction_weight
            The weight on reconstruction error (float, positive). Default: 1.
        dropout_prob
            Dropout probability of nodes (float). Default: 0
        TensorBoard
            Whether output training details through Tensorboard (bool). Default: False. Under development.
        save_model
            Whether to save trained models. Default: False. Under development.

        Return
        --------
        After training, a trained_model attribute will be added.

        Examples
        --------
        >>> from scar import model
        >>> scarObj = model(adata.X.to_df(), empty_profile)
        >>> scarObj.train()
        >>> scarObj.inference()
        >>> adata.layers["X_scar_denoised"] = scarObj.native_counts
        >>> adata.obsm["X_scar_assignment"] = scarObj.feature_assignment   # in 'cropseq' mode

        """

        list_IDs = list(range(self.raw_count.shape[0]))
        train_IDs, test_IDs = train_test_split(list_IDs, train_size=train_size)

        # Generators
        training_set = UMIDataset(self.raw_count, self.empty_profile, train_IDs)
        training_generator = torch.utils.data.DataLoader(
            training_set, batch_size=batch_size, shuffle=shuffle
        )

        val_set = UMIDataset(self.raw_count, self.empty_profile, test_IDs)
        val_generator = torch.utils.data.DataLoader(
            val_set, batch_size=batch_size, shuffle=shuffle
        )

        self.n_batch_train = len(training_generator)
        self.n_batch_val = len(val_generator)
        self.batch_size = batch_size

        # TensorBoard writer
        if TensorBoard:
            writer = SummaryWriter(TensorBoard)
            writer.add_text(
                "Experiment description",
                f"NN_layer1={self.NN_layer1}, NN_layer2={self.NN_layer2}, latent_space={self.latent_space}, kld_weight={kld_weight}, lr={lr}, epochs={epochs}, reconstruction_weight={reconstruction_weight}, dropout_prob={dropout_prob}",
                0,
            )

        # Define model
        VAE_model = VAE(
            self.num_input_feature,
            self.NN_layer1,
            self.NN_layer2,
            self.latent_space,
            self.scRNAseq_tech,
            dropout_prob,
            model=self.model,
        ).to(self.device)

        # Define optimizer
        optim = torch.optim.Adam(VAE_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, step_size=lr_step_size, gamma=lr_gamma
        )

        print("......kld_weight: ", kld_weight)
        print("......lr: ", lr)
        print("......lr_step_size: ", lr_step_size)
        print("......lr_gamma: ", lr_gamma)

        # Run training
        print("===========================================\n  Training.....")
        training_start_time = time.time()
        with std_out_err_redirect_tqdm() as orig_stdout:

            for epoch in tqdm(
                range(epochs), file=orig_stdout, dynamic_ncols=True
            ):  # tqdm needs the original stdout and dynamic_ncols=True to autodetect console width

                ################################################################################
                # Training
                ################################################################################
                train_tot_loss = 0
                train_kld_loss = 0
                train_recon_loss = 0

                VAE_model.train()
                for x_batch, ambient_freq in training_generator:

                    optim.zero_grad()
                    z, dec_nr, dec_prob, mu, var, dec_dp = VAE_model(x_batch)
                    recon_loss_minibatch, kld_loss_minibatch, loss_minibatch = loss_fn(
                        x_batch,
                        dec_nr,
                        dec_prob,
                        mu,
                        var,
                        ambient_freq,
                        reconstruction_weight=reconstruction_weight,
                        kld_weight=kld_weight,
                        dec_dp=dec_dp,
                        model=self.model,
                    )
                    loss_minibatch.backward()
                    optim.step()

                    train_tot_loss += loss_minibatch.detach().item()
                    train_recon_loss += recon_loss_minibatch.detach().item()
                    train_kld_loss += kld_loss_minibatch.detach().item()

                scheduler.step()

                # ...log the running training loss
                if TensorBoard:
                    writer.add_scalar("TrainingLoss/total loss", train_tot_loss, epoch)
                    writer.add_scalar(
                        "TrainingLoss/reconstruction loss", train_recon_loss, epoch
                    )
                    writer.add_scalar("TrainingLoss/kld_loss", train_kld_loss, epoch)
                    writer.add_scalar(
                        "learning rate", optim.param_groups[0]["lr"], epoch
                    )
                    # writer.flush()

                # ...log the running validation loss
                if TensorBoard:
                    ###################################################################################
                    # model evaluation
                    ###################################################################################
                    val_tot_loss = 0
                    val_kld_loss = 0
                    val_recon_loss = 0

                    VAE_model.eval()
                    for x_batch_val, ambient_freq_val in val_generator:

                        (
                            z_val,
                            dec_nr_val,
                            dec_prob_val,
                            mu_val,
                            var_val,
                            dec_dp_val,
                        ) = VAE_model(x_batch_val)
                        (
                            recon_loss_minibatch,
                            kld_loss_minibatch,
                            loss_minibatch,
                        ) = loss_fn(
                            x_batch_val,
                            dec_nr_val,
                            dec_prob_val,
                            mu_val,
                            var_val,
                            ambient_freq_val,
                            reconstruction_weight=reconstruction_weight,
                            kld_weight=kld_weight,
                            dec_dp=dec_dp_val,
                            model=self.model,
                        )
                        val_tot_loss += loss_minibatch.detach().item()
                        val_recon_loss += recon_loss_minibatch.detach().item()
                        val_kld_loss += kld_loss_minibatch.detach().item()

                    writer.add_scalar("ValLoss/total loss", val_tot_loss, epoch)
                    writer.add_scalar(
                        "ValLoss/reconstruction loss", val_recon_loss, epoch
                    )
                    writer.add_scalar("ValLoss/kld_loss", val_kld_loss, epoch)
                    writer.flush()

        if save_model:
            torch.save(VAE_model, save_model)

        if TensorBoard:
            writer.add_hparams(
                {
                    "lr": lr,
                    "NN_layer1": self.NN_layer1,
                    "NN_layer2": self.NN_layer2,
                    "latent_space": self.latent_space,
                    "reconstruction_weight": reconstruction_weight,
                    "kld_weight": kld_weight,
                    "epochs": epochs,
                }
            )
            writer.close()

        self.trained_model = VAE_model
        self.runtime = time.time() - training_start_time

    # Inference
    @torch.no_grad()
    def inference(
        self,
        batch_size=None,
        model="poisson",
        adjust="micro",
        feature_type="sgRNAs",
        cutoff=3,
        MOI=None,
    ):
        """
        Infering the expected native signals, noise ratios, Bayesfactors, and expected native frequencies
        Parameters
        ----------
        batch_size
            Batch_size (int). Set a value upon GPU memory issue. Default: None.
        model
            Inference model for evaluation of ambient presence (str). Default: poisson.
        adjust
            Only used for calculating Bayesfactors to improve performance. One of the following:
                'micro' -- adjust the estimated native counts per cell. This can overcome the issue of over- or under-estimation of noise. Default.
                'global' -- adjust the estimated native counts globally. This can overcome the issue of over- or under-estimation of noise.
                False -- no adjustment, use the model-returned native counts.
        feature_type
            Feature types (string), e.g., 'sgRNAs', 'CMOs', 'Tags', and etc..
        cutoff
            Cutoff for Bayesfactors. Default: 3. See https://doi.org/10.1007/s42113-019-00070-x.
        MOI(Under development)
            Multiplicity of Infection. If assigned, it will allow optimized thresholding, which tests a series of cutoffs to find the best one based on distributions of infections under given MOI. See http://dx.doi.org/10.1016/j.cell.2016.11.038. Under development.

        Return
        --------
        After inferring, several attributes will be added, inc. native_counts, bayesfactor, native_frequencies, and noise_ratio. a feature_assignment will be added in 'CROPseq' mode.

        Examples
        --------
        >>> from scar import model
        >>> scarObj = model(adata.X.to_df(), empty_profile)
        >>> scarObj.train()
        >>> scarObj.inference()
        >>> adata.layers["X_scar_denoised"] = scarObj.native_counts
        >>> adata.obsm["X_scar_assignment"] = scarObj.feature_assignment  # in 'cropseq' mode

        """

        print("===========================================\n  Inferring .....")
        total_set = UMIDataset(self.raw_count, self.empty_profile)
        num_input_feature = self.num_input_feature
        sample_size = self.raw_count.shape[0]
        self.native_counts = np.empty([sample_size, num_input_feature])
        self.bayesfactor = np.empty([sample_size, num_input_feature])
        self.native_frequencies = np.empty([sample_size, num_input_feature])
        self.noise_ratio = np.empty([sample_size, 1])

        if not batch_size:
            batch_size = sample_size
        i = 0
        self.total_generator = torch.utils.data.DataLoader(
            total_set, batch_size=batch_size, shuffle=False
        )

        for x_batch_tot, ambient_freq_tot in self.total_generator:

            minibatch_size = x_batch_tot.shape[
                0
            ]  # if not last batch, equals to batch size

            (
                native_counts_batch,
                bayesfactor_batch,
                native_frequencies_batch,
                noise_ratio_batch,
            ) = self.trained_model.inference(
                x_batch_tot, ambient_freq_tot[0, :], model=model, adjust=adjust
            )
            self.native_counts[
                i * batch_size : i * batch_size + minibatch_size, :
            ] = native_counts_batch
            self.bayesfactor[
                i * batch_size : i * batch_size + minibatch_size, :
            ] = bayesfactor_batch
            self.native_frequencies[
                i * batch_size : i * batch_size + minibatch_size, :
            ] = native_frequencies_batch
            self.noise_ratio[
                i * batch_size : i * batch_size + minibatch_size, :
            ] = noise_ratio_batch
            i += 1

        if self.scRNAseq_tech.lower() == "cropseq":
            self.assignment(feature_type=feature_type, cutoff=cutoff, MOI=MOI)
        else:
            self.feature_assignment = None

    def assignment(self, feature_type="sgRNAs", cutoff=3, MOI=None):
        """assignment of feature barcodes"""

        feature_assignment = pd.DataFrame(
            index=self.cellID, columns=[feature_type, f"n_{feature_type}"]
        )
        bayesfactor_df = pd.DataFrame(
            self.bayesfactor, index=self.cellID, columns=self.feature_names
        )
        bayesfactor_df[bayesfactor_df < cutoff] = 0  # Apply the cutoff for Bayesfactors

        for cell, row in bayesfactor_df.iterrows():
            bayesfactor_max = row[row == row.max()]
            if row.max() == 0:
                feature_assignment.loc[cell, f"n_{feature_type}"] = 0
                feature_assignment.loc[cell, feature_type] = np.nan
            elif len(bayesfactor_max) == 1:
                feature_assignment.loc[cell, f"n_{feature_type}"] = 1
                feature_assignment.loc[cell, feature_type] = bayesfactor_max.index[0]
            else:
                feature_assignment.loc[cell, f"n_{feature_type}"] = len(bayesfactor_max)
                feature_assignment.loc[cell, feature_type] = (", ").join(
                    bayesfactor_max.index
                )

        self.feature_assignment = feature_assignment


class UMIDataset(torch.utils.data.Dataset):
    """Characterizes dataset for PyTorch"""

    def __init__(self, raw_count, empty_profile, list_IDs=None):
        """Initialization"""
        self.raw_count = raw_count
        self.empty_profile = empty_profile
        if list_IDs:
            self.list_IDs = list_IDs
        else:
            self.list_IDs = list(range(raw_count.shape[0]))

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.list_IDs)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        ID = self.list_IDs[index]
        X1 = self.raw_count[ID, :]
        X2 = self.empty_profile[ID, :]
        return X1, X2
    