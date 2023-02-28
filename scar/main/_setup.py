from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData
import torch
from torch.distributions.multinomial import Multinomial


def setup_anndata(
    adata: AnnData,
    raw_adata: AnnData,
    feature_type: Union[str, list] = None,
    prob: float = 0.995,
    min_raw_counts: int = 2,
    iterations: int = 3,
    n_batch: int = 1,
    sample: int = 50000,
    kneeplot: bool = True,
    verbose: bool = True,
    figsize: tuple = (6, 6),
):
    """Calculate ambient profile for relevant features

    Identify the cell-free droplets through a multinomial distribution. See EmptyDrops [Lun2019]_ for details.


    Parameters
    ----------
    adata : AnnData
        A filtered adata object, loaded from filtered_feature_bc_matrix using `scanpy.read` , gene filtering is recommended to save memory
    raw_adata : AnnData
        An raw adata object, loaded from raw_feature_bc_matrix using `scanpy.read`
    feature_type : Union[str, list], optional
        Feature type, e.g. 'Gene Expression', 'Antibody Capture', 'CRISPR Guide Capture' or 'Multiplexing Capture', all feature types are calculated if None, by default None
    prob : float, optional
        The probability of each gene, considered as containing ambient RNA if greater than prob (joint prob euqals to the product of all genes for a droplet), by default 0.995
    min_raw_counts : int, optional
        Total counts filter for raw_adata, filtering out low counts to save memory, by default 2
    iterations : int, optional
        Total iterations, by default 3
    n_batch : int, optional
        Total number of batches, set it to a bigger number when out of memory issue occurs, by default 1
    sample : int, optional
        Randomly sample droplets to test, if greater than total droplets, use all droplets, by default 50000
    kneeplot : bool, optional
        Kneeplot to show subpopulations of droplets, by default True
    verbose : bool, optional
        Whether to display message
    figsize : tuple, optimal
        Figure size, by default (6, 6)

    Returns
    -------
    The relevant ambient profile is added in `adata.uns`

    Examples
    ---------
    .. plot::
        :context: close-figs

        import scanpy as sc
        from scar import setup_anndata
        # read filtered data
        adata = sc.read_10x_h5(filename='500_hgmm_3p_LT_Chromium_Controller_filtered_feature_bc_matrix.h5ad',
                             backup_url='https://cf.10xgenomics.com/samples/cell-exp/6.1.0/500_hgmm_3p_LT_Chromium_Controller/500_hgmm_3p_LT_Chromium_Controller_filtered_feature_bc_matrix.h5');
        adata.var_names_make_unique();
        # read raw data
        adata_raw = sc.read_10x_h5(filename='500_hgmm_3p_LT_Chromium_Controller_raw_feature_bc_matrix.h5ad',
                             backup_url='https://cf.10xgenomics.com/samples/cell-exp/6.1.0/500_hgmm_3p_LT_Chromium_Controller/500_hgmm_3p_LT_Chromium_Controller_raw_feature_bc_matrix.h5');
        adata_raw.var_names_make_unique();
        # gene and cell filter
        sc.pp.filter_genes(adata, min_counts=200);
        sc.pp.filter_genes(adata, max_counts=6000);
        sc.pp.filter_cells(adata, min_genes=200);
        # setup anndata
        setup_anndata(
            adata,
            adata_raw,
            feature_type = "Gene Expression",
            prob = 0.975,
            min_raw_counts = 2,
            kneeplot = True,
        )
    """

    if feature_type is None:
        feature_type = adata.var["feature_types"].unique()
    elif isinstance(feature_type, str):
        feature_type = [feature_type]

    # take subset genes to save memory
    # raw_adata._inplace_subset_var(raw_adata.var_names.isin(adata.var_names))
    # raw_adata._inplace_subset_obs(raw_adata.X.sum(axis=1) >= min_raw_counts)
    raw_adata = raw_adata[:, raw_adata.var_names.isin(adata.var_names)]
    raw_adata = raw_adata[raw_adata.X.sum(axis=1) >= min_raw_counts]

    raw_adata.obs["total_counts"] = raw_adata.X.sum(axis=1)

    sample = int(sample)
    idx = np.random.choice(
        raw_adata.shape[0], size=min(raw_adata.shape[0], sample), replace=False
    )
    raw_adata = raw_adata[idx]
    if verbose:
        print(
            "Randomly sample ",
            sample,
            " droplets to calculate the ambient profile.",
        )
    # initial estimation of ambient profile, will be update
    ambient_prof = raw_adata.X.sum(axis=0) / raw_adata.X.sum()

    if verbose:
        print("Estimating ambient profile for ", feature_type, "...")

    i = 0
    while i < iterations:
        # calculate joint probability (log) of being cell-free droplets for each droplet
        log_prob = []
        batch_idx = np.floor(
            np.array(range(raw_adata.shape[0])) / raw_adata.shape[0] * n_batch
        )
        for b in range(n_batch):
            try:
                count_batch = raw_adata[batch_idx == b].X.astype(int).A
            except MemoryError:
                raise MemoryError("use more batches by setting a higher n_batch")
            log_prob_batch = Multinomial(
                probs=torch.tensor(ambient_prof), validate_args=False
            ).log_prob(torch.Tensor(count_batch))
            log_prob.append(log_prob_batch)

        log_prob = np.concatenate(log_prob, axis=0)
        raw_adata.obs["log_prob"] = log_prob
        raw_adata.obs["droplets"] = "other droplets"

        # cell-containing droplets
        raw_adata.obs.loc[
            raw_adata.obs_names.isin(adata.obs_names), "droplets"
        ] = "cells"

        # identify cell-free droplets
        raw_adata.obs["droplets"] = raw_adata.obs["droplets"].mask(
            raw_adata.obs["log_prob"] >= np.log(prob) * raw_adata.shape[1],
            "cell-free droplets",
        )
        emptydrops = raw_adata[raw_adata.obs["droplets"] == "cell-free droplets"]

        if emptydrops.shape[0] < 50:
            raise Exception("Too few emptydroplets! Lower the prob parameter")

        ambient_prof = emptydrops.X.sum(axis=0) / emptydrops.X.sum()

        i += 1

        if verbose:
            print("iteration: ", i)

    # update ambient profile for each feature type
    for ft in feature_type:
        tmp = emptydrops[:, emptydrops.var["feature_types"] == ft]
        adata.uns[f"ambient_profile_{ft}"] = pd.DataFrame(
            tmp.X.sum(axis=0).reshape(-1, 1) / tmp.X.sum(),
            index=tmp.var_names,
            columns=[f"ambient_profile_{ft}"],
        )

    # update ambient profile for all feature types
    adata.uns[f"ambient_profile_{ft}"] = pd.DataFrame(
        emptydrops.X.sum(axis=0).reshape(-1, 1) / emptydrops.X.sum(),
        index=emptydrops.var_names,
        columns=[f"ambient_profile_all"],
    )

    if kneeplot:
        _, axs = plt.subplots(2, figsize=figsize)

        all_droplets = raw_adata.obs.copy()
        all_droplets = (
            all_droplets.sort_values(by="total_counts", ascending=False)
            .reset_index()
            .rename_axis("rank_by_counts")
            .reset_index()
        )
        all_droplets = all_droplets.loc[all_droplets["total_counts"] >= min_raw_counts]
        all_droplets = all_droplets.set_index("index").rename_axis("cells")
        all_droplets = (
            all_droplets.sort_values(by="log_prob", ascending=True)
            .reset_index()
            .rename_axis("rank_by_log_prob")
            .reset_index()
            .set_index("cells")
        )

        ax = sns.lineplot(
            data=all_droplets,
            x="rank_by_counts",
            y="total_counts",
            hue="droplets",
            hue_order=["cells", "other droplets", "cell-free droplets"],
            palette=sns.color_palette()[-3:],
            markers=False,
            lw=2,
            ci=None,
            ax=axs[0],
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("")
        ax.set_title("cell-free droplets have lower counts")

        all_droplets["prob"] = np.exp(all_droplets["log_prob"])
        ax = sns.lineplot(
            data=all_droplets,
            x="rank_by_log_prob",
            y="prob",
            hue="droplets",
            hue_order=["cells", "other droplets", "cell-free droplets"],
            palette=sns.color_palette()[-3:],
            markers=False,
            lw=2,
            ci=None,
            ax=axs[1],
        )
        ax.set_xscale("log")
        ax.set_xlabel("sorted droplets")
        ax.set_title("cell-free droplets have relatively higher probs")

        plt.tight_layout()
