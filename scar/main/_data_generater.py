# -*- coding: utf-8 -*-
"""Module to generate synthetic datasets with ambient contamination"""

import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

###################################################################################
# Synthetic scrnaseq datasets


class scrnaseq:
    """Generate synthetic single-cell RNAseq data with ambient contamination

    Parameters
    ----------
    n_cells : int
        number of cells
    n_celltypes : int
        number of cell types
    n_features : int
        number of features (mRNA)
    n_total_molecules : int, optional
        total molecules per cell, by default 8000
    capture_rate : float, optional
        the probability of being captured by beads, by default 0.7

    Examples
    --------
    .. plot::
        :context: close-figs

        import numpy as np
        from scar import data_generator

        n_features = 1000  # 1000 genes, bad visualization with too big number
        n_cells = 6000  # cells
        n_total_molecules = 20000 # total mRNAs
        n_celltypes = 8  # cell types

        np.random.seed(8)
        scRNAseq = data_generator.scrnaseq(n_cells, n_celltypes, n_features, n_total_molecules=n_total_molecules)
        scRNAseq.generate(dirichlet_concentration_hyper=1)
        scRNAseq.heatmap(vmax=5)

    """

    def __init__(
        self, n_cells, n_celltypes, n_features, n_total_molecules=8000, capture_rate=0.7
    ):
        """initilization"""
        self.n_cells = n_cells
        """int, number of cells"""
        self.n_celltypes = n_celltypes
        """int, number of cell types"""
        self.n_features = n_features
        """int, number of features (mRNA, sgRNA, ADT, tag, CMO, and etc.)"""
        self.n_total_molecules = n_total_molecules
        """int, number of total molecules per cell"""
        self.capture_rate = capture_rate
        """float, the probability of being captured by beads"""
        self.obs_count = (None,)
        """vector, observed counts"""
        self.ambient_profile = (None,)
        """vector, the probability of occurrence of each ambient transcript"""
        self.cell_identity = (None,)
        """matrix, the onehot expression of the identity of cell types"""
        self.noise_ratio = (None,)
        """vector, contamination level per cell"""
        self.celltype = (None,)
        """vector, the identity of cell types"""
        self.ambient_signals = (None,)
        """matrix, the real ambient signals"""
        self.native_signals = (None,)
        """matrix, the real native signals"""
        self.native_profile = (None,)
        """matrix, the frequencies of the real native signals"""
        self.total_counts = (None,)
        """vector, the total observed counts per cell"""
        self.empty_droplets = None
        """matrix, synthetic cell-free droplets"""

    def generate(self, dirichlet_concentration_hyper=0.05):
        """Generate a synthetic scRNAseq dataset.

        Parameters
        ----------
        dirichlet_concentration_hyper : None or real, optional, \
            the concentration hyperparameters of dirichlet distribution. \
                Determining the sparsity of native signals. \
                    If None, 1 / n_features, by default 0.005.
    
        Returns
        -------
            After running, several attributes are added
        """

        if dirichlet_concentration_hyper:
            alpha = np.ones(self.n_features) * dirichlet_concentration_hyper
        else:
            alpha = np.ones(self.n_features) / self.n_features

        # simulate native expression frequencies for each cell
        cell_comp_prior = random.dirichlet(np.ones(self.n_celltypes))
        celltype = random.choice(
            a=self.n_celltypes, size=self.n_cells, p=cell_comp_prior
        )
        cell_identity = np.identity(self.n_celltypes)[celltype]
        theta_celltype = random.dirichlet(alpha, size=self.n_celltypes)

        beta_in_each_cell = cell_identity.dot(theta_celltype)

        # simulate total molecules for a droplet in ambient pool
        n_total_mol = random.randint(
            low=self.n_total_molecules / 5, high=self.n_total_molecules / 2, size=1
        )

        # simulate ambient signals
        beta0 = random.dirichlet(np.ones(self.n_features))
        tot_count0 = random.negative_binomial(
            n_total_mol, self.capture_rate, size=self.n_cells
        )
        ambient_signals = np.vstack(
            [random.multinomial(n=tot_c, pvals=beta0) for tot_c in tot_count0]
        )

        # add empty droplets
        tot_count0_empty = random.negative_binomial(
            n_total_mol, self.capture_rate, size=self.n_cells
        )
        ambient_signals_empty = np.vstack(
            [random.multinomial(n=tot_c, pvals=beta0) for tot_c in tot_count0_empty]
        )

        # simulate native signals
        tot_trails = random.randint(
            low=self.n_total_molecules / 2,
            high=self.n_total_molecules,
            size=self.n_celltypes,
        )
        tot_count1 = [
            random.negative_binomial(tot, self.capture_rate)
            for tot in cell_identity.dot(tot_trails)
        ]

        native_signals = np.vstack(
            [
                random.multinomial(n=tot_c, pvals=theta1)
                for tot_c, theta1 in zip(tot_count1, beta_in_each_cell)
            ]
        )
        obs = ambient_signals + native_signals

        noise_ratio = tot_count0 / (tot_count0 + tot_count1)

        self.obs_count = obs
        self.ambient_profile = beta0
        self.cell_identity = cell_identity
        self.noise_ratio = noise_ratio
        self.celltype = celltype
        self.ambient_signals = ambient_signals
        self.native_signals = native_signals
        self.native_profile = beta_in_each_cell
        self.total_counts = obs.sum(axis=1)
        self.empty_droplets = ambient_signals_empty.astype(int)

    def heatmap(
        self, feature_type="mRNA", return_obj=False, figsize=(12, 4), vmin=0, vmax=10
    ):
        """Heatmap of synthetic data.

        Parameters
        ----------
        feature_type : str, optional
            the feature types, by default "mRNA"
        return_obj : bool, optional
            whether to output figure object, by default False
        figsize : tuple, optional
            figure size, by default (15, 5)
        vmin : int, optional
            colorbar minimum, by default 0
        vmax : int, optional
            colorbar maximum, by default 10

        Returns
        -------
        fig object
            if return_obj, return a fig object
        """
        sort_cell_idx = []
        for f in self.ambient_profile.argsort():
            sort_cell_idx += list(np.where(self.celltype == f)[0])

        native_signals = self.native_signals[sort_cell_idx][
            :, self.ambient_profile.argsort()
        ]
        ambient_signals = self.ambient_signals[sort_cell_idx][
            :, self.ambient_profile.argsort()
        ]
        obs = self.obs_count[sort_cell_idx][:, self.ambient_profile.argsort()]

        fig, axs = plt.subplots(ncols=3, figsize=figsize)
        sns.heatmap(
            np.log2(obs + 1),
            yticklabels=False,
            vmin=vmin,
            vmax=vmax,
            cmap="coolwarm",
            center=1,
            ax=axs[0],
            rasterized=True,
            cbar_kws={"label": "log2(counts + 1)"},
        )
        axs[0].set_title("noisy observation")

        sns.heatmap(
            np.log2(ambient_signals + 1),
            yticklabels=False,
            vmin=vmin,
            vmax=vmax,
            cmap="coolwarm",
            center=1,
            ax=axs[1],
            rasterized=True,
            cbar_kws={"label": "log2(counts + 1)"},
        )
        axs[1].set_title("ambient signals")

        sns.heatmap(
            np.log2(native_signals + 1),
            yticklabels=False,
            vmin=vmin,
            vmax=vmax,
            cmap="coolwarm",
            center=1,
            ax=axs[2],
            rasterized=True,
            cbar_kws={"label": "log2(counts + 1)"},
        )
        axs[2].set_title("native signals")

        fig.supxlabel(feature_type)
        fig.supylabel("cells")
        plt.tight_layout()

        if return_obj:
            return fig


######################################################################################
# Synthetic citeseq datasets
class citeseq(scrnaseq):
    """Generate synthetic ADT count data for CITE-seq with ambient contamination.

    Parameters
    ----------
    n_cells : int
        number of cells
    n_celltypes : int
        number of cell types
    n_features : int
        number of distinct antibodies (ADTs)
    n_total_molecules : int, optional
        number of total molecules, by default 8000
    capture_rate : float, optional
        the probabilities of being captured by beads, by default 0.7

    Examples
    --------
    .. plot::
        :context: close-figs

        import numpy as np
        from scar import data_generator

        n_features = 50  # 50 ADTs
        n_cells = 6000  # 6000 cells
        n_celltypes = 6  # cell types

        # generate a synthetic ADT count dataset
        np.random.seed(8)
        citeseq = data_generator.citeseq(n_cells, n_celltypes, n_features)
        citeseq.generate()
        citeseq.heatmap()

    """

    def __init__(
        self, n_cells, n_celltypes, n_features, n_total_molecules=8000, capture_rate=0.7
    ):
        super().__init__(
            n_cells, n_celltypes, n_features, n_total_molecules, capture_rate
        )

    def generate(self, dirichlet_concentration_hyper=None):
        """Generate a synthetic ADT dataset.

        Parameters
        ----------
        dirichlet_concentration_hyper : None or real, optional \
            the concentration hyperparameters of dirichlet distribution. \
                If None, 1 / n_features, by default None
    
        Returns
        -------
            After running, several attributes are added
        """

        if dirichlet_concentration_hyper:
            alpha = np.ones(self.n_features) * dirichlet_concentration_hyper
        else:
            alpha = np.ones(self.n_features) / self.n_features

        # simulate native expression frequencies for each cell
        cell_comp_prior = random.dirichlet(np.ones(self.n_celltypes))
        celltype = random.choice(
            a=self.n_celltypes, size=self.n_cells, p=cell_comp_prior
        )
        cell_identity = np.identity(self.n_celltypes)[celltype]
        theta_celltype = random.dirichlet(alpha, size=self.n_celltypes)
        beta_in_each_cell = cell_identity.dot(theta_celltype)

        # simulate total molecules for a droplet in ambient pool
        n_total_mol = random.randint(
            low=self.n_total_molecules / 5, high=self.n_total_molecules / 2, size=1
        )

        # simulate ambient signals
        beta0 = random.dirichlet(np.ones(self.n_features))
        tot_count0 = random.negative_binomial(
            n_total_mol, self.capture_rate, size=self.n_cells
        )
        ambient_signals = np.vstack(
            [random.multinomial(n=tot_c, pvals=beta0) for tot_c in tot_count0]
        )

        # add empty droplets
        tot_count0_empty = random.negative_binomial(
            n_total_mol, self.capture_rate, size=self.n_cells
        )
        ambient_signals_empty = np.vstack(
            [random.multinomial(n=tot_c, pvals=beta0) for tot_c in tot_count0_empty]
        )

        # simulate native signals
        tot_trails = random.randint(
            low=self.n_total_molecules / 2,
            high=self.n_total_molecules,
            size=self.n_celltypes,
        )
        tot_count1 = [
            random.negative_binomial(tot, self.capture_rate)
            for tot in cell_identity.dot(tot_trails)
        ]

        native_signals = np.vstack(
            [
                random.multinomial(n=tot_c, pvals=theta1)
                for tot_c, theta1 in zip(tot_count1, beta_in_each_cell)
            ]
        )
        obs = ambient_signals + native_signals

        noise_ratio = tot_count0 / (tot_count0 + tot_count1)

        self.obs_count = obs
        self.ambient_profile = beta0
        self.cell_identity = cell_identity
        self.noise_ratio = noise_ratio
        self.celltype = celltype
        self.ambient_signals = ambient_signals
        self.native_signals = native_signals
        self.native_profile = beta_in_each_cell
        self.total_counts = obs.sum(axis=1)
        self.empty_droplets = ambient_signals_empty.astype(int)

    def heatmap(
        self, feature_type="ADT", return_obj=False, figsize=(12, 4), vmin=0, vmax=10
    ):
        """Heatmap of synthetic data.

        Parameters
        ----------
        feature_type : str, optional
            the feature types, by default "ADT"
        return_obj : bool, optional
            whether to output figure object, by default False
        figsize : tuple, optional
            figure size, by default (15, 5)
        vmin : int, optional
            colorbar minimum, by default 0
        vmax : int, optional
            colorbar maximum, by default 10

        Returns
        -------
        fig object
            if return_obj, return a fig object
        """
        sort_cell_idx = []
        for f in self.ambient_profile.argsort():
            sort_cell_idx += list(np.where(self.celltype == f)[0])

        native_signals = self.native_signals[sort_cell_idx][
            :, self.ambient_profile.argsort()
        ]
        ambient_signals = self.ambient_signals[sort_cell_idx][
            :, self.ambient_profile.argsort()
        ]
        obs = self.obs_count[sort_cell_idx][:, self.ambient_profile.argsort()]

        fig, axs = plt.subplots(ncols=3, figsize=figsize)
        sns.heatmap(
            np.log2(obs + 1),
            yticklabels=False,
            vmin=vmin,
            vmax=vmax,
            cmap="coolwarm",
            center=1,
            ax=axs[0],
            rasterized=True,
            cbar_kws={"label": "log2(counts + 1)"},
        )
        axs[0].set_title("noisy observation")

        sns.heatmap(
            np.log2(ambient_signals + 1),
            yticklabels=False,
            vmin=vmin,
            vmax=vmax,
            cmap="coolwarm",
            center=1,
            ax=axs[1],
            rasterized=True,
            cbar_kws={"label": "log2(counts + 1)"},
        )
        axs[1].set_title("ambient signals")

        sns.heatmap(
            np.log2(native_signals + 1),
            yticklabels=False,
            vmin=vmin,
            vmax=vmax,
            cmap="coolwarm",
            center=1,
            ax=axs[2],
            rasterized=True,
            cbar_kws={"label": "log2(counts + 1)"},
        )
        axs[2].set_title("native signals")

        fig.supxlabel(feature_type)
        fig.supylabel("cells")
        plt.tight_layout()

        if return_obj:
            return fig


##########################################################################################
# Synthetic cropseq datasets


class cropseq(scrnaseq):
    """Generate synthetic sgRNA count data for scCRISPRseq with ambient contamination

    Parameters
    ----------
    n_cells : int
        number of cells
    n_celltypes : int
        number of cell types
    n_features : int
        number of dinstinct sgRNAs
    library_pattern : str, optional
        the pattern of sgRNA libraries, three possibilities: \
            "uniform" - each sgRNA has equal frequency in the libraries. \
            "pyramid" - a few sgRNAs have significantly higher frequencies in the libraries. \
            "reverse_pyramid" - a few sgRNAs have significantly lower frequencies in the libraries.\
            by default "pyramid".
    noise_ratio : float, optional
        global contamination level, by default 0.005
    average_counts_per_cell : int, optional
        average total sgRNA counts per cell, by default 2000
    doublet_rate : int, optional
        doublet rate, by default 0
    missing_rate : int, optional
        the fraction of droplets which have zero sgRNAs integrated, by default 0

    Examples
    --------
    
    .. plot::
        :context: close-figs
        
        import numpy as np
        from scar import data_generator
        
        n_features = 100  # 100 sgRNAs in the libraries
        n_cells = 6000  # 6000 cells
        n_celltypes = 1  # single cell line
        
        # generate a synthetic sgRNA count dataset
        np.random.seed(8)
        cropseq = data_generator.cropseq(n_cells, n_celltypes, n_features)
        cropseq.generate(noise_ratio=0.98)
        cropseq.heatmap(vmax=6)
    """

    def __init__(
        self,
        n_cells,
        n_celltypes,
        n_features,
    ):
        super().__init__(n_cells, n_celltypes, n_features)

        self.sgrna_freq = None
        """vector, sgRNA frequencies in the libraries
        """

    # generate a pool of sgrnas
    def _set_sgrna_frequency(self):
        """set the pattern of sgrna library"""
        if self.library_pattern == "uniform":
            self.sgrna_freq = 1.0 / self.n_features
        elif self.library_pattern == "pyramid":
            uniform_spaced_values = np.random.permutation(self.n_features + 1) / (
                self.n_features + 1
            )
            uniform_spaced_values = uniform_spaced_values[uniform_spaced_values != 0]
            log_values = np.log(uniform_spaced_values)
            self.sgrna_freq = log_values / np.sum(log_values)
        elif self.library_pattern == "reverse_pyramid":
            uniform_spaced_values = (
                np.random.permutation(self.n_features) / self.n_features
            )
            log_values = (uniform_spaced_values + 0.001) ** (1 / 10)
            self.sgrna_freq = log_values / np.sum(log_values)

    def _set_native_signals(self):
        """generatation of native signals"""

        self._set_sgrna_frequency()

        # cells without any sgrnas
        n_cells_miss = int(self.n_cells * self.missing_rate)

        # Doublets
        n_doublets = int(self.n_cells * self.doublet_rate)

        # total number of single sgrnas which are integrated into cells
        # (cells with double sgrnas will be counted twice)
        n_cells_integrated = self.n_cells - n_cells_miss + n_doublets

        # create cells with sgrnas based on sgrna frequencies
        self.celltype = random.choice(
            a=range(self.n_features), size=n_cells_integrated, p=self.sgrna_freq
        )
        self.cell_identity = np.eye(self.n_features)[self.celltype]  # cell_identity

    def _add_ambient(self):
        """add ambient signals"""
        self._set_native_signals()
        sgrna_mixed_freq = (
            1 - self.noise_ratio
        ) * self.cell_identity + self.noise_ratio * self.sgrna_freq
        sgrna_mixed_freq = sgrna_mixed_freq / sgrna_mixed_freq.sum(
            axis=1, keepdims=True
        )
        return sgrna_mixed_freq

    # function to generate counts per cell
    def generate(
        self,
        dirichlet_concentration_hyper=None,
        library_pattern="pyramid",
        noise_ratio=0.96,
        average_counts_per_cell=2000,
        doublet_rate=0,
        missing_rate=0,
    ):
        """Generate a synthetic sgRNA count dataset.

        Parameters
        ----------
        library_pattern : str, optional
            library pattern, by default "pyramid"
        noise_ratio : float, optional
            global contamination level, by default 0.005
        average_counts_per_cell : int, optional
            average total sgRNA counts per cell, by default 2000
        doublet_rate : int, optional
            doublet rate, by default 0
        missing_rate : int, optional
            the fraction of droplets which have zero sgRNAs integrated, by default 0

        Returns
        -------
            After running, several attributes are added
        """

        assert library_pattern.lower() in ["uniform", "pyramid", "reverse_pyramid"]
        self.library_pattern = library_pattern.lower()
        """str, library pattern
        """
        self.doublet_rate = doublet_rate
        """float, doublet rate
        """
        self.missing_rate = missing_rate
        """float, the fraction of droplets which have zero sgRNAs integrated.
        """
        self.noise_ratio = noise_ratio
        """float, global contamination level
        """
        self.average_counts_per_cell = average_counts_per_cell
        """int, the mean of total sgRNA counts per cell
        """

        # generate total counts per cell
        total_counts = random.negative_binomial(
            self.average_counts_per_cell, 0.7, size=self.n_cells
        )

        # the mixed sgrna expression profile
        sgrna_mixed_freq = self._add_ambient()

        # final count matrix:
        obs = np.vstack(
            [
                random.multinomial(n=tot_c, pvals=p)
                for tot_c, p in zip(total_counts, sgrna_mixed_freq)
            ]
        )

        self.obs_count = obs
        self.total_counts = total_counts
        self.ambient_profile = self.sgrna_freq
        self.native_signals = (
            self.total_counts.reshape(-1, 1)
            * (1 - self.noise_ratio)
            * self.cell_identity
        )
        self.ambient_signals = np.clip(obs - self.native_signals, 0, None)

    def heatmap(
        self, feature_type="sgRNAs", return_obj=False, figsize=(12, 4), vmin=0, vmax=7
    ):
        """Heatmap of synthetic data.

        Parameters
        ----------
        feature_type : str, optional
            the feature types, by default "sgRNAs"
        return_obj : bool, optional
            whether to output figure object, by default False
        figsize : tuple, optional
            figure size, by default (15, 5)
        vmin : int, optional
            colorbar minimum, by default 0
        vmax : int, optional
            colorbar maximum, by default 10

        Returns
        -------
        fig object
            if return_obj, return a fig object
        """
        sort_cell_idx = []
        for f in self.ambient_profile.argsort():
            sort_cell_idx += list(np.where(self.celltype == f)[0])

        native_signals = self.native_signals[sort_cell_idx][
            :, self.ambient_profile.argsort()
        ]
        ambient_signals = self.ambient_signals[sort_cell_idx][
            :, self.ambient_profile.argsort()
        ]
        obs = self.obs_count[sort_cell_idx][:, self.ambient_profile.argsort()]

        fig, axs = plt.subplots(ncols=3, figsize=figsize)
        sns.heatmap(
            np.log2(obs + 1),
            yticklabels=False,
            vmin=vmin,
            vmax=vmax,
            cmap="coolwarm",
            center=1,
            ax=axs[0],
            rasterized=True,
            cbar_kws={"label": "log2(counts + 1)"},
        )
        axs[0].set_title("noisy observation")

        sns.heatmap(
            np.log2(ambient_signals + 1),
            yticklabels=False,
            vmin=vmin,
            vmax=vmax,
            cmap="coolwarm",
            center=1,
            ax=axs[1],
            rasterized=True,
            cbar_kws={"label": "log2(counts + 1)"},
        )
        axs[1].set_title("ambient signals")

        sns.heatmap(
            np.log2(native_signals + 1),
            yticklabels=False,
            vmin=vmin,
            vmax=vmax,
            cmap="coolwarm",
            center=1,
            ax=axs[2],
            rasterized=True,
            cbar_kws={"label": "log2(counts + 1)"},
        )
        axs[2].set_title("native signals")

        fig.supxlabel(feature_type)
        fig.supylabel("cells")
        plt.tight_layout()

        if return_obj:
            return fig
