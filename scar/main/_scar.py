# -*- coding: utf-8 -*-

"""The main module of scar"""

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
    """The scar model

        Parameters
        ----------
        raw_count : Union[str, np.ndarray, pd.DataFrame]
            Raw count matrix.
        ambient_profile : Optional[Union[str, np.ndarray, pd.DataFrame]], optional
            the probability of occurrence of each ambient transcript.\
                If None, averaging cells to estimate the ambient profile, by default None
        nn_layer1 : int, optional
            number of neurons of the 1st layer, by default 150
        nn_layer2 : int, optional
            number of neurons of the 2nd layer, by default 100
        latent_dim : int, optional
            number of neurons of the bottleneck layer, by default 15
        dropout_prob : float, optional
            dropout probability of neurons, by default 0
        feature_type : str, optional
            the feature to be denoised. One of the following:  

                | 'mRNA' -- transcriptome
                | 'ADT' -- protein counts in CITE-seq
                | 'sgRNA' -- sgRNA counts for scCRISPRseq
                | 'tag' -- identity barcodes or any data types of super high sparsity. \
                    E.g., in cell indexing experiments, we would expect a single true signal \
                        (1) and many negative signals (0) for each cell
                | 'CMO' -- Cell Multiplexing Oligo
                | By default "mRNA"
        count_model : str, optional
            the model to generate the UMI count. One of the following:  

                | 'binomial' -- binomial model,
                | 'poisson' -- poisson model,
                | 'zeroinflatedpoisson' -- zeroinflatedpoisson model, by default "binomial"
        sparsity : float, optional
            range: [0, 1]. The sparsity of expected native signals. \
            It varies between datasets, e.g. if one prefilters genes -- \
                use only highly variable genes -- \
                    the sparsity should be low; on the other hand, it should be set high \
                        in the case of unflitered genes. \
                        Forced to be one in the mode of "sgRNA(s)" and "tag(s)". \
                            Thank Will Macnair for the valuable feedback.

        Raises
        ------
        TypeError
            if raw_count is not str or np.ndarray or pd.DataFrame
        TypeError
            if ambient_profile is not str or np.ndarray or pd.DataFrame or None

        Examples
        --------
            >>> # Real data
            >>> import scanpy as sc
            >>> from scar import model
            >>> adata = sc.read("...")  # load an anndata object
            >>> scarObj = model(adata.X.to_df(), ambient_profile)  # initialize scar model
            >>> scarObj.train()  # start training
            >>> scarObj.inference()  # inference
            >>> adata.layers["X_scar_denoised"] = scarObj.native_counts   # results are saved in scarObj
            >>> adata.obsm["X_scar_assignment"] = scarObj.feature_assignment   #'sgRNA' or 'tag' feature type

        Examples
        -------------------------
        .. plot::
            :context: close-figs

            # Synthetic data
            import numpy as np
            import seaborn as sns
            import matplotlib.pyplot as plt
            from scar import data_generator, model

            # Generate a synthetic ADT count dataset
            np.random.seed(8)
            n_features = 50  # 50 ADTs
            n_cells = 6000  # 6000 cells
            n_celltypes = 6  # cell types
            citeseq = data_generator.citeseq(n_cells, n_celltypes, n_features)
            citeseq.generate()
            
            # Train scAR
            citeseq_denoised = model(citeseq.obs_count, citeseq.ambient_profile, feature_type="ADT", sparsity=0.6)  # initialize scar model
            citeseq_denoised.train(epochs=100, verbose=False)  # start training
            citeseq_denoised.inference()  # inference

            # Visualization
            sorted_noisy_counts = citeseq.obs_count[citeseq.celltype.argsort()][
                        :, citeseq.ambient_profile.argsort()
                    ]  # noisy observation
            sorted_native_counts = citeseq.native_signals[citeseq.celltype.argsort()][
                        :, citeseq.ambient_profile.argsort()
                    ]  # native counts
            sorted_denoised_counts = citeseq_denoised.native_counts[citeseq.celltype.argsort()][
                        :, citeseq.ambient_profile.argsort()
                    ]  # denoised counts

            fig, axs = plt.subplots(ncols=3, figsize=(12,4))
            sns.heatmap(
                        np.log2(sorted_noisy_counts + 1),
                        yticklabels=False,
                        vmin=0,
                        vmax=10,
                        cmap="coolwarm",
                        center=1,
                        ax=axs[0],
                        cbar_kws={"label": "log2(counts + 1)"},
                    )
            axs[0].set_title("noisy observation")

            sns.heatmap(
                        np.log2(sorted_native_counts + 1),
                        yticklabels=False,
                        vmin=0,
                        vmax=10,
                        cmap="coolwarm",
                        center=1,
                        ax=axs[1],
                        cbar_kws={"label": "log2(counts + 1)"},
                    )
            axs[1].set_title("native counts (ground truth)")

            sns.heatmap(
                        np.log2(sorted_denoised_counts + 1),
                        yticklabels=False,
                        vmin=0,
                        vmax=10,
                        cmap="coolwarm",
                        center=1,
                        ax=axs[2],
                        cbar_kws={"label": "log2(counts + 1)"},
                    )
            axs[2].set_title("denoised counts (prediction)")
            
            fig.supxlabel("ADTs")
            fig.supylabel("cells")
            plt.tight_layout()
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
        sparsity: float = .9
    ):
        """initialize object"""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        """str, "cuda" if gpu is available
        """        
        self.nn_layer1 = nn_layer1
        """int, number of neurons of the 1st layer.
        """
        self.nn_layer2 = nn_layer2
        """int, number of neurons of the 2nd layer.
        """        
        self.latent_dim = latent_dim
        """int, number of neurons of the bottleneck layer.
        """
        self.dropout_prob = dropout_prob
        """float, dropout probability of neurons.
        """
        self.feature_type = feature_type
        """str, the feature to be denoised. One of the following:

            | 'mRNA' -- transcriptome
            | 'ADT' -- protein counts in CITE-seq
            | 'sgRNA' -- sgRNA counts for scCRISPRseq
            | 'tag' -- identity barcodes or any data types of super high sparsity. \
                E.g., in cell indexing experiments, we would expect a single true signal \
                    (1) and many negative signals (0) for each cell.
            | 'CMO' -- Cell Multiplexing Oligo
            | By default "mRNA"
        """
        self.count_model = count_model
        """str, the model to generate the UMI count. One of the following:  

            | 'binomial' -- binomial model,
            | 'poisson' -- poisson model,
            | 'zeroinflatedpoisson' -- zeroinflatedpoisson model.
        """
        self.sparsity = sparsity
        """float, the sparsity of expected native signals. (0, 1]. \
            Forced to be one in the mode of "sgRNA(s)" and "tag(s)".
        """

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
        """raw_count : np.ndarray, raw count matrix.
        """
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
        """ambient_profile : np.ndarray, the probability of occurrence of each ambient transcript.
        """
        
        # self.n_batch_train = None
        # self.n_batch_val = None
        # self.batch_size = None
        self.runtime = None
        """int, runtime in seconds.
        """
        self.trained_model = None
        """nn.Module object, added after training.
        """
        self.native_counts = None
        """np.ndarray, denoised counts, added after inference
        """
        self.bayesfactor = None
        """np.ndarray, bayesian factor of whether native signals are present, added after inference
        """
        self.native_frequencies = None
        """np.ndarray, probability of native transcripts (normalized denoised counts), added after inference
        """
        self.noise_ratio = None
        """np.ndarray, noise ratio per cell, added after inference
        """
        self.feature_assignment = None
        """pd.DataFrame, assignment of sgRNA or tag or other feature barcodes, added after inference or assignment
        """

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
        """train training scar model

        Parameters
        ----------
        batch_size : int, optional
            batch size, by default 64
        train_size : float, optional
            the size of training samples, by default 0.998
        shuffle : bool, optional
            whether to shuffle the data, by default True
        kld_weight : float, optional
            weight of KL loss, by default 1e-5
        lr : float, optional
            initial learning rate, by default 1e-3
        lr_step_size : int, optional
            `period of learning rate decay, \
                <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html/>`_,\
                    by default 5
        lr_gamma : float, optional
            multiplicative factor of learning rate decay, by default 0.97
        epochs : int, optional
            training iterations, by default 800
        reconstruction_weight : float, optional
            weight on reconstruction error, by default 1
        dropout_prob : float, optional
            dropout probability of neurons, by default 0
        TensorBoard : bool, optional
            whether to output training details through Tensorboard \
                (under development), by default False
        save_model : bool, optional
            whether to save trained models(under development), by default False
        verbose : bool, optional
            whether to print the details, by default True       
        Returns
        -------
            After training, a trained_model attribute will be added.       
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

        # self.n_batch_train = len(training_generator)
        # self.n_batch_val = len(val_generator)
        # self.batch_size = batch_size

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
            sparsity=self.sparsity,
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
        self, batch_size=None, count_model_inf="poisson", adjust="micro", cutoff=3, round_to_int="stochastic_rounding", moi=None
    ):
        """inference infering the expected native signals, noise ratios, Bayesfactors and expected native frequencies

        Parameters
        ----------
        batch_size : int, optional
            batch size, set a small value upon GPU memory issue, by default None
        count_model_inf : str, optional
            inference model for evaluation of ambient presence, by default "poisson"
        adjust : str, optional
            Only used for calculating Bayesfactors to improve performance. \
                One of the following:  

                    | 'micro' -- adjust the estimated native counts per cell. \
                    This can overcome the issue of over- or under-estimation of noise.
                    | 'global' -- adjust the estimated native counts globally. \
                    This can overcome the issue of over- or under-estimation of noise.
                    | False -- no adjustment, use the model-returned native counts.
                    | Defaults to "micro"
        cutoff : int, optional
            cutoff for Bayesfactors, by default 3
        round_to_int : str, optional
            whether to round the counts, by default "stochastic_rounding"
        moi : int, optional (under development)
            multiplicity of infection. If assigned, it will allow optimized thresholding, \
                which tests a series of cutoffs to find the best one \
                    based on distributions of infections under given moi.\
                        See Perturb-seq [Dixit2016]_ for details, by default None
        Returns
        -------
            After inferring, several attributes will be added, inc. native_counts, bayesfactor,\
            native_frequencies, and noise_ratio. \
                A feature_assignment will be added in 'sgRNA' or 'tag' or 'CMO' feature type.       
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
                round_to_int=round_to_int
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

        if self.feature_type.lower() in ["sgrna", "sgrnas", "tag", "tags", "cmo", "cmos"]:
            self.assignment(cutoff=cutoff, moi=moi)
        else:
            self.feature_assignment = None

    def assignment(self, cutoff=3, moi=None):
        """assignment assignment of feature barcodes. Re-run it can test different cutoffs for your experiments.

        Parameters
        ----------
        cutoff : int, optional
            cutoff for Bayesfactors, by default 3
        moi : float, optional
            multiplicity of infection. (under development)\
            If assigned, it will allow optimized thresholding,\
                which tests a series of cutoffs to find the best one \
                    based on distributions of infections under given moi.\
                        See  Perturb-seq [Dixit2016]_, by default None
        Returns
        -------
            After running, a attribute 'feature_assignment' will be added,\
                in 'sgRNA' or 'tag' or 'CMO' feature type.       
        Raises
        ------
        NotImplementedError
            if moi is not None
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
                feature_assignment.loc[cell, self.feature_type] = ""
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
                    bayesfactor_max.index.astype(str)
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
      