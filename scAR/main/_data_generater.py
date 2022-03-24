"""Data generater"""
# -*- coding: utf-8 -*-

import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

# Class of synthetic scRNAseq datasets
class scRNAseq_synthetic:
    """
    Class to generate synthetic scRNAseq/CITE-seq data, which contain ambient signals.

    Parameters
    ----------
    n_cells
        The number of cells.
    n_celltypes
        The number of cell types or cells with distinct identity barcodes.
    n_features
        The number of features, e.g., genes or sgRNAs or cell indexing barcodes \
        or antibody-conjugated ologos.
    n_total_molecules
        The total real number of molecules in a cell. Together with the following \
        capture_rate, it defines the total molecules being observed in a cell. \
        By default, 4000.
    capture_rate
        The capture rate for each molecule. Together with the above total number of \
        molecules, it defines the total observed molecules in a cell. \
        By default, 0.7.
    """

    def __init__(self, n_cells, n_celltypes, n_features, n_total_molecules=8000, capture_rate=0.7):
        self.n_cells = n_cells
        self.n_celltypes = n_celltypes
        self.n_features = n_features
        self.n_total_molecules = n_total_molecules
        self.capture_rate = capture_rate


# Synthetic CITEseq datasets
class CITEseq(scRNAseq_synthetic):
    """Synthetic CITEseq datasets"""
    def __init__(self, n_cells, n_celltypes, n_features, n_total_molecules=8000, capture_rate=0.7):
        super().__init__(n_cells, n_celltypes, n_features, n_total_molecules, capture_rate)

    def generate(self):
        """generate"""
        # simulate native expression frequencies for each cell
        cell_comp_prior = random.dirichlet(np.ones(self.n_celltypes))
        celltype = random.choice(a=self.n_celltypes, size=self.n_cells, p=cell_comp_prior)
        cell_identity = np.identity(self.n_celltypes)[celltype]
        theta_celltype = random.dirichlet(
            np.ones(self.n_features) / self.n_features, size=self.n_celltypes
        )
        beta_in_each_cell = cell_identity.dot(theta_celltype)

        # simulate total molecules for a droplet in ambient pool
        n_total_mol = random.randint(
            low=self.n_total_molecules / 5, high=self.n_total_molecules / 2, size=1
        )

        # simulate ambient signals
        beta0 = random.dirichlet(np.ones(self.n_features))
        tot_count0 = random.negative_binomial(n_total_mol, self.capture_rate, size=self.n_cells)
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
            low=self.n_total_molecules / 2, high=self.n_total_molecules, size=self.n_celltypes
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

    def heatmap(self, return_obj=False):
        """heatmap"""
        native_signals = self.native_signals[self.celltype.argsort()]
        ambient_signals = self.ambient_signals[self.celltype.argsort()]
        obs = self.obs_count[self.celltype.argsort()]

        fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
        sns.heatmap(
            np.log2(obs + 1),
            yticklabels=False,
            vmin=0,
            vmax=10,
            cmap="coolwarm",
            center=2,
            ax=axs[0],
        )
        axs[0].set_title("observation")

        sns.heatmap(
            np.log2(ambient_signals + 1),
            yticklabels=False,
            vmin=0,
            vmax=10,
            cmap="coolwarm",
            center=2,
            ax=axs[1],
        )
        axs[1].set_title("ambient signals")

        sns.heatmap(
            np.log2(native_signals + 1),
            yticklabels=False,
            vmin=0,
            vmax=10,
            cmap="coolwarm",
            center=2,
            ax=axs[2],
        )
        axs[2].set_title("native signals")

        fig.supxlabel("protein markers")
        fig.supylabel("cells")
        plt.tight_layout()

        if return_obj:
            return fig


# Synthetic CROPseq datasets
class CROPseq(scRNAseq_synthetic):
    """
    Create synthetic data of CROPseq
    """

    def __init__(
        self,
        n_cells,
        n_celltypes,
        n_features,
        library_pattern="pyramid",
        noise_ratio=0.005,
        average_counts_per_cell=2000,
        doublet_rate=0,
        missing_rate=0,
    ):
        super().__init__(n_cells, n_celltypes, n_features)
        assert library_pattern.lower() in ["uniform", "pyramid", "reverse_pyramid"]
        self.library_pattern = library_pattern.lower()
        self.doublet_rate = doublet_rate
        self.missing_rate = missing_rate
        self.noise_ratio = noise_ratio
        self.average_counts_per_cell = average_counts_per_cell

    def set_sgRNA_frequency(self):
        """generate a pool of sgRNAs"""
        if self.library_pattern == "uniform":
            self.sgRNA_freq = 1.0 / self.n_features
        elif self.library_pattern == "pyramid":
            uniform_spaced_values = np.random.permutation(self.n_features + 1) / (
                self.n_features + 1
            )
            uniform_spaced_values = uniform_spaced_values[uniform_spaced_values != 0]
            log_values = np.log(uniform_spaced_values)
            self.sgRNA_freq = log_values / np.sum(log_values)
        elif self.library_pattern == "reverse_pyramid":
            uniform_spaced_values = np.random.permutation(self.n_features) / self.n_features
            log_values = (uniform_spaced_values + 0.001) ** (1 / 10)
            self.sgRNA_freq = log_values / np.sum(log_values)

    def set_native_signals(self):
        """generate native signal"""
        self.set_sgRNA_frequency()

        # cells without any sgRNAs
        n_cells_miss = int(self.n_cells * self.missing_rate)

        # Doublets
        n_doublets = int(self.n_cells * self.doublet_rate)

        # total number of single sgRNAs which are integrated into cells
        # (cells with double sgRNAs will be counted twice)
        n_cells_integrated = self.n_cells - n_cells_miss + n_doublets

        # create cells with sgRNAs based on sgRNA frequencies
        self.celltype = random.choice(
            a=range(self.n_features), size=n_cells_integrated, p=self.sgRNA_freq
        )
        self.cell_identity = np.eye(self.n_features)[self.celltype]  # cell_identity

    def add_ambient(self):
        """add_ambient"""
        self.set_native_signals()
        sgRNA_mixed_freq = (
            1 - self.noise_ratio
        ) * self.cell_identity + self.noise_ratio * self.sgRNA_freq
        sgRNA_mixed_freq = sgRNA_mixed_freq / sgRNA_mixed_freq.sum(axis=1, keepdims=True)
        return sgRNA_mixed_freq

    def generate(self):
        """generate counts per cell"""
        # generate total counts per cell
        total_counts = random.negative_binomial(
            self.average_counts_per_cell, 0.7, size=self.n_cells
        )

        # the mixed sgRNA expression profile
        sgRNA_mixed_freq = self.add_ambient()

        # final count matrix:
        obs = np.vstack(
            [
                random.multinomial(n=tot_c, pvals=p)
                for tot_c, p in zip(total_counts, sgRNA_mixed_freq)
            ]
        )

        self.obs_count = obs
        self.total_counts = total_counts
        self.ambient_profile = self.sgRNA_freq
        self.native_signals = (
            self.total_counts.reshape(-1, 1) * (1 - self.noise_ratio) * self.cell_identity
        )
        self.ambient_signals = np.clip(obs - self.native_signals, 0, None)

    def heatmap(self, return_obj=False):
        """heatmap"""
        native_signals = self.native_signals[self.celltype.argsort()]
        ambient_signals = self.ambient_signals[self.celltype.argsort()]
        obs = self.obs_count[self.celltype.argsort()]

        fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
        sns.heatmap(
            np.log2(obs + 1),
            yticklabels=False,
            vmin=0,
            vmax=7,
            cmap="coolwarm",
            center=2,
            ax=axs[0],
        )
        axs[0].set_title("observation")

        sns.heatmap(
            np.log2(ambient_signals + 1),
            yticklabels=False,
            vmin=0,
            vmax=7,
            cmap="coolwarm",
            center=2,
            ax=axs[1],
        )
        axs[1].set_title("ambient signals")

        sns.heatmap(
            np.log2(native_signals + 1),
            yticklabels=False,
            vmin=0,
            vmax=7,
            cmap="coolwarm",
            center=2,
            ax=axs[2],
        )
        axs[2].set_title("native signals")

        fig.supxlabel("sgRNAs")
        fig.supylabel("cells")
        plt.tight_layout()

        if return_obj:
            return fig
