# -*- coding: utf-8 -*-

"""The main module of scar"""

import sys, time, contextlib, torch
from typing import Optional, Union
from scipy import sparse
import numpy as np, pandas as pd, anndata as ad

from torch.utils.data import Dataset, random_split, DataLoader
from tqdm import tqdm
from tqdm.contrib import DummyTqdmFile

from ._vae import VAE
from ._loss_functions import loss_fn
from ._utils import get_logger


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
        raw_count : Union[str, np.ndarray, pd.DataFrame, ad.AnnData]
            Raw count matrix or Anndata object.

            .. note::
               scar takes the raw UMI counts as input. No size normalization or log transformation.

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

                | 'mRNA' -- transcriptome data, including scRNAseq and snRNAseq
                | 'ADT' -- protein counts in CITE-seq
                | 'sgRNA' -- sgRNA counts for scCRISPRseq
                | 'tag' -- identity barcodes or any data types of high sparsity. \
                    E.g., in cell indexing experiments, we would expect a single true signal \
                        (1) and many negative signals (0) for each cell
                | 'CMO' -- Cell Multiplexing Oligo counts for cell hashing
                | 'ATAC' -- peak counts for scATACseq
                .. versionadded:: 0.5.2
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
        
            .. versionadded:: 0.4.0
        cache_capacity : int, optional
            the capacity of caching data on GPU. Set a smaller value upon GPU memory issue. By default 20000 cells are cached.

            .. versionadded:: 0.7.0
        batch_key : str, optional
            batch key in AnnData.obs, by default None. \
                If assigned, batch ambient removel will be performed and \
                the ambient profile will be estimated for each batch.

            .. versionadded:: 0.7.0

        device : str, optional
            either "auto, "cpu" or "cuda" or "mps", by default "auto"
        verbose : bool, optional
            whether to print the details, by default True

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
            >>> scarObj = model(adata, ambient_profile)  # initialize scar model
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
            sorted_denoised_counts = citeseq_denoised.native_counts.toarray()[citeseq.celltype.argsort()][
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
        raw_count: Union[str, np.ndarray, pd.DataFrame, ad.AnnData],
        ambient_profile: Optional[Union[str, np.ndarray, pd.DataFrame]] = None,
        nn_layer1: int = 150,
        nn_layer2: int = 100,
        latent_dim: int = 15,
        dropout_prob: float = 0,
        feature_type: str = "mRNA",
        count_model: str = "binomial",
        sparsity: float = 0.9,
        batch_key: str = None,
        device: str = "auto",
        cache_capacity: int = 20000,
        verbose: bool = True,
    ):
        """initialize object"""

        self.logger = get_logger("model", verbose=verbose)
        """logging.Logger, the logger for this class.
        """

        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.logger.info(f"{self.device} is detected and will be used.")
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = torch.device("mps")
                self.logger.info(f"{self.device} is detected and will be used.")
            else:
                self.device = torch.device("cpu")
                self.logger.info(f"No GPU detected. {self.device} will be used.")
        else:
            self.device = device
            self.logger.info(f"{device} will be used.")

        """str, either "auto, "cpu" or "cuda".
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
            | 'CMO' -- Cell Multiplexing Oligo counts for cell hashing
            | 'ATAC' -- peak counts for scATACseq           
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
        self.cache_capacity = cache_capacity
        """int, the capacity of caching data on GPU. Set a smaller value upon GPU memory issue. By default 20000 cells are cached on GPU/MPS.

            .. versionadded:: 0.7.0
        """
        
        if isinstance(raw_count, ad.AnnData):
            if batch_key is not None:
                if batch_key not in raw_count.obs.columns:
                    raise ValueError(f"{batch_key} not found in AnnData.obs.")
                
                self.logger.info(
                    f"Found {raw_count.obs[batch_key].nunique()} batches defined by {batch_key} in AnnData.obs. Estimating ambient profile per batch..."
                )
                batch_id_per_cell = pd.Categorical(raw_count.obs[batch_key]).codes
                ambient_profile = np.empty((len(np.unique(batch_id_per_cell)),raw_count.shape[1]))
                for batch_id in np.unique(batch_id_per_cell):
                    subset = raw_count[batch_id_per_cell==batch_id]
                    ambient_profile[batch_id, :] = subset.X.sum(axis=0) / subset.X.sum()

                # add a mapper to locate the batch id
                self.batch_id = batch_id_per_cell
                self.n_batch = len(np.unique(batch_id_per_cell))
            else:
                # get ambient profile from AnnData.uns
                if "ambient_profile_all" in raw_count.uns:
                    self.logger.info(
                        "Found ambient profile in AnnData.uns['ambient_profile_all']"
                    )
                    ambient_profile = raw_count.uns["ambient_profile_all"]
                else:
                    self.logger.info(
                        "Ambient profile not found in AnnData.uns['ambient_profile'], estimating it by averaging pooled cells..."
                    )

        elif isinstance(raw_count, str):
            # read pickle file into dataframe
            raw_count = pd.read_pickle(raw_count)

        elif isinstance(raw_count, np.ndarray):
            # convert np.array to pd.DataFrame
            raw_count = pd.DataFrame(
                raw_count,
                index=range(raw_count.shape[0]),
                columns=range(raw_count.shape[1]),
            )

        elif isinstance(raw_count, pd.DataFrame):
            pass
        else:
            raise TypeError(
                f"Expecting str or np.array or pd.DataFrame or AnnData object, but get a {type(raw_count)}"
            )

        self.raw_count = raw_count
        """raw_count : np.ndarray, raw count matrix.
        """
        self.n_features = raw_count.shape[1]
        """int, number of features.
        """
        self.cell_id = raw_count.index.to_list() if isinstance(raw_count, pd.DataFrame) else raw_count.obs_names.to_list()
        """list, cell id.
        """
        self.feature_names = raw_count.columns.to_list() if isinstance(raw_count, pd.DataFrame) else raw_count.var_names.to_list()
        """list, feature names.
        """

        if isinstance(ambient_profile, str):
            ambient_profile = pd.read_pickle(ambient_profile)
            ambient_profile = ambient_profile.fillna(0).values  # missing vals -> zeros
        elif isinstance(ambient_profile, pd.DataFrame):
            ambient_profile = ambient_profile.fillna(0).values  # missing vals -> zeros
        elif isinstance(ambient_profile, np.ndarray):
            ambient_profile = np.nan_to_num(ambient_profile)  # missing vals -> zeros
        elif not ambient_profile:
            self.logger.info(" Evaluate ambient profile from cells")
            if isinstance(raw_count, pd.DataFrame):
                ambient_profile = raw_count.sum() / raw_count.sum().sum()
                ambient_profile = ambient_profile.fillna(0).values
            elif isinstance(raw_count, ad.AnnData):
                ambient_profile = np.array(raw_count.X.sum(axis=0)/raw_count.X.sum())
                ambient_profile = np.nan_to_num(ambient_profile).flatten()
        else:
            raise TypeError(
                f"Expecting str / np.array / None / pd.DataFrame, but get a {type(ambient_profile)}"
            )

        if ambient_profile.squeeze().ndim == 1:
            ambient_profile = (
                ambient_profile.squeeze()
                .reshape(1, -1)
            )
            # add a mapper to locate the artificial batch id
            self.batch_id = np.zeros(raw_count.shape[0], dtype=int)#.reshape(-1, 1)
            self.n_batch = 1

        self.ambient_profile = ambient_profile
        """ambient_profile : np.ndarray, the probability of occurrence of each ambient transcript.
        """

        self.runtime = None
        """int, runtime in seconds.
        """
        self.loss_values = None
        """list, loss values during training.
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
        epochs: int = 400,
        reconstruction_weight: float = 1,
        dropout_prob: float = 0,
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
                <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html/>`_\
                    by default 5
        lr_gamma : float, optional
            multiplicative factor of learning rate decay, by default 0.97
        epochs : int, optional
            training iterations, by default 800
        reconstruction_weight : float, optional
            weight on reconstruction error, by default 1
        dropout_prob : float, optional
            dropout probability of neurons, by default 0
        save_model : bool, optional
            whether to save trained models(under development), by default False
        verbose : bool, optional
            whether to print the details, by default True       
        Returns
        -------
            After training, a trained_model attribute will be added.
               
        """
        # Generators
        total_dataset = UMIDataset(self.raw_count, self.ambient_profile, self.batch_id, device=self.device, cache_capacity=self.cache_capacity)
        training_set, validation_set = random_split(total_dataset, [train_size, 1 - train_size])
        training_generator = DataLoader(
            training_set, batch_size=batch_size, shuffle=shuffle,
            drop_last=True
        )
        self.dataset = total_dataset

        loss_values = []

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
            n_batch=self.n_batch,
            verbose=verbose,
        ).to(self.device)
        # Define optimizer
        optim = torch.optim.Adam(vae_nets.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, step_size=lr_step_size, gamma=lr_gamma
        )

        self.logger.info(f"kld_weight: {kld_weight:.2e}")
        self.logger.info(f"learning rate: {lr:.2e}")
        self.logger.info(f"lr_step_size: {lr_step_size:d}")
        self.logger.info(f"lr_gamma: {lr_gamma:.2f}")

        # Run training
        training_start_time = time.time()
        with std_out_err_redirect_tqdm() as orig_stdout:
            # Initialize progress bar
            progress_bar = tqdm(
                total=epochs,
                file=orig_stdout,
                dynamic_ncols=True,
                desc="Training",
            )
            progress_bar.clear()
            for _ in range(epochs):
                train_tot_loss = 0
                train_kld_loss = 0
                train_recon_loss = 0

                vae_nets.train()
                for x_batch, ambient_freq, batch_id_onehot in training_generator:
                    optim.zero_grad()
                    dec_nr, dec_prob, means, var, dec_dp = vae_nets(x_batch, batch_id_onehot)
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

                avg_train_tot_loss = train_tot_loss / len(training_generator)
                loss_values.append(avg_train_tot_loss)

                progress_bar.set_postfix({"Loss": "{:.4e}".format(avg_train_tot_loss)})
                progress_bar.update()

            progress_bar.close()

        if save_model:
            torch.save(vae_nets, save_model)

        self.loss_values = loss_values
        self.trained_model = vae_nets
        self.runtime = time.time() - training_start_time

    # Inference
    @torch.no_grad()
    def inference(
        self,
        batch_size=4096,
        count_model_inf="poisson",
        adjust="micro",
        cutoff=3,
        round_to_int="stochastic_rounding",
        clip_to_obs=False,
        get_native_frequencies=False,
        moi=None,
    ):
        """inference infering the expected native signals, noise ratios, Bayesfactors and expected native frequencies

        Parameters
        ----------
        batch_size : int, optional
            batch size, set a small value upon GPU memory issue, by default 4096
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

            .. versionadded:: 0.4.1

        clip_to_obs : bool, optional
            whether to clip the predicted native counts to the observation in order to ensure \
                that denoised counts are not greater than the observation, by default False. \
                Use it with caution, as it may lead to over-estimation of overall noise.

            .. versionadded:: 0.5.0
        
        get_native_frequencies : bool, optional
            whether to get native frequencies, by default False

            .. versionadded:: 0.7.0

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
        n_features = self.n_features
        sample_size = self.raw_count.shape[0]

        dt = np.int64 if round_to_int=="stochastic_rounding" else np.float32
        native_counts = sparse.lil_matrix((sample_size, n_features), dtype=dt)
        noise_ratio = sparse.lil_matrix((sample_size, 1), dtype=np.float32)

        native_frequencies = sparse.lil_matrix((sample_size, n_features), dtype=np.float32) if get_native_frequencies else None

        if self.feature_type.lower() in [
            "sgrna",
            "sgrnas",
            "tag",
            "tags",
            "cmo",
            "cmos",
            "atac",
        ]:
            bayesfactor = sparse.lil_matrix((sample_size, n_features), dtype=np.float32)
        else:
            bayesfactor = None
        
        if not batch_size:
            batch_size = sample_size
        i = 0
        generator_full_data = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=False
        )

        for x_batch_tot, ambient_freq_tot, x_batch_id_onehot_tot in generator_full_data:
            minibatch_size = x_batch_tot.shape[
                0
            ]  # if not the last batch, equals to batch size

            (
                native_counts_batch,
                bayesfactor_batch,
                native_frequencies_batch,
                noise_ratio_batch,
            ) = self.trained_model.inference(
                x_batch_tot,
                x_batch_id_onehot_tot,
                ambient_freq_tot[0, :],
                count_model_inf=count_model_inf,
                adjust=adjust,
                round_to_int=round_to_int,
                clip_to_obs=clip_to_obs,
            )
            native_counts[
                i * batch_size : i * batch_size + minibatch_size, :
            ] = native_counts_batch
            noise_ratio[
                i * batch_size : i * batch_size + minibatch_size, :
            ] = noise_ratio_batch
            if native_frequencies is not None:
                native_frequencies[
                    i * batch_size : i * batch_size + minibatch_size, :
                ] = native_frequencies_batch
            if bayesfactor is not None:
                bayesfactor[
                    i * batch_size : i * batch_size + minibatch_size, :
                ] = bayesfactor_batch

            i += 1

        self.native_counts = native_counts.tocsr()
        self.noise_ratio = noise_ratio.tocsr()
        self.bayesfactor = bayesfactor.tocsr() if bayesfactor is not None else None
        self.native_frequencies = native_frequencies.tocsr() if native_frequencies is not None else None

        if self.feature_type.lower() in [
            "sgrna",
            "sgrnas",
            "tag",
            "tags",
            "cmo",
            "cmos",
            "atac",
        ]:
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
            self.bayesfactor.toarray(), index=self.cell_id, columns=self.feature_names
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

class UMIDataset(Dataset):
    """Characterizes dataset for PyTorch"""

    def __init__(self, raw_count, ambient_profile, batch_id, device, cache_capacity=20000):
        """Initialization"""
        
        self.raw_count = torch.from_numpy(raw_count.fillna(0).values).int() if isinstance(raw_count, pd.DataFrame) else raw_count
        self.ambient_profile = torch.from_numpy(ambient_profile).float().to(device)
        self.batch_id = torch.from_numpy(batch_id).to(torch.int64).to(device)
        self.batch_onehot = torch.from_numpy(np.eye(len(np.unique(batch_id)))).to(torch.int64).to(device)
        self.device = device
        self.cache_capacity = cache_capacity

        # Cache data
        self.cache = {}

    def __len__(self):
        """Denotes the total number of samples"""
        return self.raw_count.shape[0]

    def __getitem__(self, index):
        """Generates one sample of data"""

        if index in self.cache:
            return self.cache[index]
        else:
            # Select samples
            sc_count = self.raw_count[index].to(self.device) if isinstance(self.raw_count, torch.Tensor) else torch.from_numpy(self.raw_count[index].X.toarray().flatten()).int().to(self.device)
            sc_ambient = self.ambient_profile[self.batch_id[index], :]
            sc_batch_id_onehot = self.batch_onehot[self.batch_id[index], :]

            # Cache samples
            if len(self.cache) <= self.cache_capacity:
                self.cache[index] = (sc_count, sc_ambient, sc_batch_id_onehot)
            
            return sc_count, sc_ambient, sc_batch_id_onehot
