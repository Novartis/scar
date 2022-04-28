# -*- coding: utf-8 -*-

"""The main function of scar"""

import sys
import time
from typing import Optional, Union
import contextlib
import numpy as np
import pandas as pd

import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tqdm.contrib import DummyTqdmFile

from ._vae import VAE
from ._loss_functions import loss_fn

@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    """
    Writing progressbar into stdout rather than stderr,
    from https://github.com/tqdm/tqdm/blob/master/examples/redirect_print.py
    """
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
    """scAR class object

    Args:
        raw_count (Union[str, np.ndarray, pd.DataFrame]): Raw count matrix.
        ambient_profile (Optional[Union[str, np.ndarray, pd.DataFrame]], optional): \
            the probability of occurrence of each ambient transcript. Defaults to None.\
                If None, averaging cells to estimate the ambient profile
        nn_layer1 (int, optional): number of neurons of the 1st layer. Defaults to 150.
        nn_layer2 (int, optional): number of neurons of the 2nd layer. Defaults to 100.
        latent_dim (int, optional): number of neurons of the bottleneck layer. Defaults to 15.
        dropout_prob (float, optional): dropout probability of neurons. Defaults to 0.
        feature_type (str, optional): the feature to be denoised. One of the following:
            'mRNA' -- transcriptome.
            'ADT' -- protein counts in CITE-seq
            'sgRNA' -- sgRNA counts for scCRISPRseq
            'tag' -- identity barcodes or any data types of super high sparsity. \
                E.g., in cell indexing experiments, we would expect a single true signal \
                    (1) and many negative signals (0) for each cell. Defaults to "mRNA".
        count_model (str, optional): the model to generate the UMI count. One of the following:
            'binomial' -- binomial model. Defaults to "binomial".
            'poisson' -- poisson model
            'zeroinflatedpoisson' -- zeroinflatedpoisson model
    Examples:
        >>> # import package
        >>> import scanpy as sc
        >>> from scar import model
        >>> adata = sc.read("...")  # load an anndata object
        >>> scarObj = model(adata.X.to_df(), ambient_profile)  # initialize scar model
        >>> # see the following method for training and inference

    """

    def __init__(
        self,
        raw_count: Union[str, np.ndarray, pd.DataFrame],
        ambient_profile: Optional[Union[str, np.ndarray, pd.DataFrame]] = None,
        nn_layer1: int = 150,
        nn_layer2: int = 100,
        latent_dim: int = 15,
        dropout_prob: float = 0,
        feature_type: str = "mRNA",
        count_model: str = "binomial",
    ):
        """initialize object"""

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.nn_layer1 = nn_layer1
        self.nn_layer2 = nn_layer2
        self.latent_dim = latent_dim
        self.dropout_prob = dropout_prob
        self.feature_type = feature_type
        self.count_model = count_model

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
                f"Expecting str or np.array or pd.DataFrame object, but get a {type(raw_count)}"
            )

        raw_count = raw_count.fillna(0)  # missing vals -> zeros

        # Loading numpy to tensor on GPU
        self.raw_count = torch.from_numpy(raw_count.values).int().to(self.device)
        self.n_features = raw_count.shape[1]
        self.cell_id = list(raw_count.index)
        self.feature_names = list(raw_count.columns)

        if isinstance(ambient_profile, str):
            ambient_profile = pd.read_pickle(ambient_profile)
            ambient_profile = ambient_profile.fillna(0).values  # missing vals -> zeros
        elif isinstance(ambient_profile, pd.DataFrame):
            ambient_profile = ambient_profile.fillna(0).values  # missing vals -> zeros
        elif isinstance(ambient_profile, np.ndarray):
            ambient_profile = np.nan_to_num(ambient_profile)  # missing vals -> zeros
        elif not ambient_profile:
            print(" ... Evaluate empty profile from cells")
            ambient_profile = raw_count.sum() / raw_count.sum().sum()
            ambient_profile = ambient_profile.fillna(0).values
        else:
            raise TypeError(
                f"Expecting str / np.array / None / pd.DataFrame, but get a {type(ambient_profile)}"
            )

        if ambient_profile.squeeze().ndim == 1:
            ambient_profile = (
                ambient_profile.squeeze()
                .reshape(1, -1)
                .repeat(raw_count.shape[0], axis=0)
            )
        self.ambient_profile = torch.from_numpy(ambient_profile).float().to(self.device)

        self.n_batch_train = None
        self.n_batch_val = None
        self.batch_size = None
        self.runtime = None
        self.trained_model = None
        self.native_counts = None
        self.bayesfactor = None
        self.native_frequencies = None
        self.noise_ratio = None
        self.feature_assignment = None

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
        save_model: bool = False,
        verbose: bool = True,
    ):
        """Method for training scar model

        Args:
            batch_size (int, optional): batch size. Defaults to 64.
            train_size (float, optional): the size of training samples. Defaults to 0.998.
            shuffle (bool, optional): whether to shuffle the data. Defaults to True.
            kld_weight (float, optional): weight of KL loss. Defaults to 1e-5.
            lr (float, optional): initial learning rate. Defaults to 1e-3.
            lr_step_size (int, optional): period of learning rate decay,
                see https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html.
                Defaults to 5.
            lr_gamma (float, optional): multiplicative factor of learning rate decay.
                Defaults to 0.97.
            epochs (int, optional): training iterations. Defaults to 800.
            reconstruction_weight (float, optional): weight on reconstruction error. Defaults to 1.
            dropout_prob (float, optional): dropout probability of neurons. Defaults to 0.
            TensorBoard (bool, optional): Whether output training details through Tensorboard
                (under development). Defaults to False.
            save_model (bool, optional): whether to save trained models(under development).
                Defaults to False.
            verbose (bool, optional): whether to print the details. Defaults to True.
        Returns:
            After training, a trained_model attribute will be added.
        Examples:
            >>> # import package
            >>> import scanpy as sc
            >>> from scar import model
            >>> adata = sc.read("...")  # load an anndata object
            >>> scarObj = model(adata.X.to_df(), ambient_profile)  # initialize scar model
            >>> scarObj.train()  # start training
            >>> # all parameter should converge, we need next step to estimate the noise and etc.
        """

        list_ids = list(range(self.raw_count.shape[0]))
        train_ids, test_ids = train_test_split(list_ids, train_size=train_size)

        # Generators
        training_set = UMIDataset(self.raw_count, self.ambient_profile, train_ids)
        training_generator = torch.utils.data.DataLoader(
            training_set, batch_size=batch_size, shuffle=shuffle)
        val_set = UMIDataset(self.raw_count, self.ambient_profile, test_ids)
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
                f"nn_layer1={self.nn_layer1}"
                + f"nn_layer2={self.nn_layer2}"
                + f"latent_dim={self.latent_dim}"
                + f"kld_weight={kld_weight}"
                + f"lr={lr}"
                + f"epochs={epochs}"
                + f"reconstruction_weight={reconstruction_weight}"
                + f"dropout_prob={dropout_prob}",
                0,
            )

        # Define model
        vae_nets = VAE(
            n_features=self.n_features,
            nn_layer1=self.nn_layer1,
            nn_layer2=self.nn_layer2,
            latent_dim=self.latent_dim,
            dropout_prob=dropout_prob,
            feature_type=self.feature_type,
            count_model=self.count_model,
            verbose=verbose,
        ).to(self.device)

        # Define optimizer
        optim = torch.optim.Adam(vae_nets.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=lr_step_size, gamma=lr_gamma)        
        if verbose:
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

                vae_nets.train()
                for x_batch, ambient_freq in training_generator:

                    optim.zero_grad()
                    dec_nr, dec_prob, means, var, dec_dp = vae_nets(x_batch)
                    recon_loss_minibatch, kld_loss_minibatch, loss_minibatch = loss_fn(
                        x_batch,
                        dec_nr,
                        dec_prob,
                        means,
                        var,
                        ambient_freq,
                        reconstruction_weight=reconstruction_weight,
                        kld_weight=kld_weight,
                        dec_dp=dec_dp,
                        count_model=self.count_model,
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
                    ##################################################################
                    # model evaluation
                    ##################################################################
                    val_tot_loss = 0
                    val_kld_loss = 0
                    val_recon_loss = 0

                    vae_nets.eval()
                    for x_batch_val, ambient_freq_val in val_generator:

                        (
                            dec_nr_val,
                            dec_prob_val,
                            mu_val,
                            var_val,
                            dec_dp_val
                        ) = vae_nets(x_batch_val)

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
                            count_model=self.count_model,
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
            torch.save(vae_nets, save_model)

        # if TensorBoard:
        #     writer.add_hparams(
        #         {
        #             "lr": lr,
        #             "nn_layer1": self.nn_layer1,
        #             "nn_layer2": self.nn_layer2,
        #             "latent_dim": self.latent_dim,
        #             "reconstruction_weight": reconstruction_weight,
        #             "kld_weight": kld_weight,
        #             "epochs": epochs,
        #         }
        #     )
        #     writer.close()

        self.trained_model = vae_nets
        self.runtime = time.time() - training_start_time

    # Inference
    @torch.no_grad()
    def inference(
        self, batch_size=None, count_model_inf="poisson", adjust="micro", cutoff=3, moi=None
    ):
        """Infering the expected native signals, noise ratios, Bayesfactors and expected native frequencies

        Args:
            batch_size (_type_, optional): batch size, set a small value upon GPU memory issue.
                Defaults to None.
            count_model_inf (str, optional): Inference model for evaluation of ambient presence.
                Defaults to "poisson".
            adjust (str, optional): Only used for calculating Bayesfactors to improve performance.
                One of the following: 
                    'micro' -- adjust the estimated native counts per cell. \
                    This can overcome the issue of over- or under-estimation of noise. Default.
                    'global' -- adjust the estimated native counts globally.\
                    This can overcome the issue of over- or under-estimation of noise.
                    False -- no adjustment, use the model-returned native counts. Defaults to "micro".
            cutoff (int, optional): cutoff for Bayesfactors. Defaults to 3.
            moi (float, optional): multiplicity of infection. If assigned, it will allow optimized thresholding, 
                which tests a series of cutoffs to find the best one based on distributions of infections under given moi.
                    See http://dx.doi.org/10.1016/j.cell.2016.11.038. Under development. Defaults to None.
        Returns:
            After inferring, several attributes will be added, inc. native_counts, bayesfactor,\
            native_frequencies, and noise_ratio. A feature_assignment will be added in 'sgRNA' or 'tag' feature type.
        Examples:
            >>> # import package
            >>> import scanpy as sc
            >>> from scar import model
            >>> adata = sc.read("...")  # load an anndata object
            >>> scarObj = model(adata.X.to_df(), ambient_profile)  # initialize scar model
            >>> scarObj.train()  # start training
            >>> scarObj.inference()  # inference
            >>> adata.layers["X_scar_denoised"] = scarObj.native_counts   # results are saved in scarObj
            >>> adata.obsm["X_scar_assignment"] = scarObj.feature_assignment   #'sgRNA' or 'tag' feature type

        """

        print("===========================================\n  Inferring .....")
        total_set = UMIDataset(self.raw_count, self.ambient_profile)
        n_features = self.n_features
        sample_size = self.raw_count.shape[0]
        self.native_counts = np.empty([sample_size, n_features])
        self.bayesfactor = np.empty([sample_size, n_features])
        self.native_frequencies = np.empty([sample_size, n_features])
        self.noise_ratio = np.empty([sample_size, 1])

        if not batch_size:
            batch_size = sample_size
        i = 0
        generator_full_data = torch.utils.data.DataLoader(
            total_set, batch_size=batch_size, shuffle=False
        )

        for x_batch_tot, ambient_freq_tot in generator_full_data:

            minibatch_size = x_batch_tot.shape[
                0
            ]  # if not last batch, equals to batch size

            (
                native_counts_batch,
                bayesfactor_batch,
                native_frequencies_batch,
                noise_ratio_batch,
            ) = self.trained_model.inference(
                x_batch_tot,
                ambient_freq_tot[0, :],
                count_model_inf=count_model_inf,
                adjust=adjust,
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

        if self.feature_type.lower() in ["sgrna", "sgrnas", "tag", "tags"]:
            self.assignment(cutoff=cutoff, moi=moi)
        else:
            self.feature_assignment = None

    def assignment(self, cutoff=3, moi=None):
        """assignment of feature barcodes

        Args:
            cutoff (int, optional): cutoff for Bayesfactors. Defaults to 3.
            moi (float, optional): multiplicity of infection. If assigned, it will allow optimized thresholding, 
                which tests a series of cutoffs to find the best one based on distributions of infections under given moi.
                    See http://dx.doi.org/10.1016/j.cell.2016.11.038. Under development. Defaults to None.

        Raises:
            NotImplementedError: if moi is not None
        """

        feature_assignment = pd.DataFrame(
            index=self.cell_id, columns=[self.feature_type, f"n_{self.feature_type}"]
        )
        bayesfactor_df = pd.DataFrame(
            self.bayesfactor, index=self.cell_id, columns=self.feature_names
        )
        bayesfactor_df[bayesfactor_df < cutoff] = 0  # Apply the cutoff for Bayesfactors

        for cell, row in bayesfactor_df.iterrows():
            bayesfactor_max = row[row == row.max()]
            if row.max() == 0:
                feature_assignment.loc[cell, f"n_{self.feature_type}"] = 0
                feature_assignment.loc[cell, self.feature_type] = np.nan
            elif len(bayesfactor_max) == 1:
                feature_assignment.loc[cell, f"n_{self.feature_type}"] = 1
                feature_assignment.loc[cell, self.feature_type] = bayesfactor_max.index[
                    0
                ]
            else:
                feature_assignment.loc[cell, f"n_{self.feature_type}"] = len(
                    bayesfactor_max
                )
                feature_assignment.loc[cell, self.feature_type] = (", ").join(
                    bayesfactor_max.index
                )

        self.feature_assignment = feature_assignment

        if moi:
            raise NotImplementedError

class UMIDataset(torch.utils.data.Dataset):
    """Characterizes dataset for PyTorch"""

    def __init__(self, raw_count, ambient_profile, list_ids=None):
        """Initialization"""
        self.raw_count = raw_count
        self.ambient_profile = ambient_profile
        if list_ids:
            self.list_ids = list_ids
        else:
            self.list_ids = list(range(raw_count.shape[0]))

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.list_ids)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        sc_id = self.list_ids[index]
        sc_count = self.raw_count[sc_id, :]
        sc_ambient = self.ambient_profile[sc_id, :]
        return sc_count, sc_ambient
      