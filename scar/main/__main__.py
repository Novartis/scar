# -*- coding: utf-8 -*-
"""command line of scar"""

import argparse

import os
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix
from ._scar import model
from ..__init__ import __version__
from ._utils import get_logger


def main():
    """main function for command line interface"""
    args = Config()
    count_matrix_path = args.count_matrix[0]
    ambient_profile_path = args.ambient_profile
    feature_type = args.feature_type
    output_dir = (
        os.getcwd() if not args.output else args.output
    )  # if None, output to current directory
    count_model = args.count_model
    nn_layer1 = args.hidden_layer1
    nn_layer2 = args.hidden_layer2
    latent_dim = args.latent_dim
    epochs = args.epochs
    device = args.device
    sparsity = args.sparsity
    save_model = args.save_model
    batch_size = args.batchsize
    batch_size_infer = args.batchsize_infer
    adjust = args.adjust
    cutoff = args.cutoff
    moi = args.moi
    round_to_int = args.round2int
    clip_to_obs = args.clip_to_obs
    verbose = args.verbose

    main_logger = get_logger("scar", verbose=verbose)

    _, file_extension = os.path.splitext(count_matrix_path)

    if file_extension == ".pickle":
        count_matrix = pd.read_pickle(count_matrix_path)

    # Denoising transcritomic data
    elif file_extension == ".h5":
        adata = sc.read_10x_h5(count_matrix_path, gex_only=False)

        main_logger.info(
            "unprocessed data contains: {0} cells and {1} genes".format(
                adata.shape[0], adata.shape[1]
            )
        )
        adata = adata[:, adata.X.sum(axis=0) > 0]  # filter out features of zero counts
        main_logger.info(
            "filter out features of zero counts, remaining data contains: {0} cells and {1} genes".format(
                adata.shape[0], adata.shape[1]
            )
        )

        if feature_type.lower() == "all":
            features = adata.var["feature_types"].unique()
            count_matrix = adata.to_df()

        # Denoising mRNAs
        elif feature_type.lower() in ["mrna", "mrnas"]:
            features = "Gene Expression"
            adata_fb = adata[:, adata.var["feature_types"] == features]
            count_matrix = adata_fb.to_df()

        # Denoising sgRNAs
        elif feature_type.lower() in ["sgrna", "sgrnas"]:
            features = "CRISPR Guide Capture"
            adata_fb = adata[:, adata.var["feature_types"] == features]
            count_matrix = adata_fb.to_df()

        # Denoising CMO tags
        elif feature_type.lower() in ["tag", "tags"]:
            features = "Multiplexing Capture"
            adata_fb = adata[:, adata.var["feature_types"] == features]
            count_matrix = adata_fb.to_df()

        # Denoising ADTs
        elif feature_type.lower() in ["adt", "adts"]:
            features = "Antibody Capture"
            adata_fb = adata[:, adata.var["feature_types"] == features]
            count_matrix = adata_fb.to_df()

        # Denoising ATAC peaks
        elif feature_type.lower() in ["atac"]:
            features = "Peaks"
            adata_fb = adata[:, adata.var["feature_types"] == features]
            count_matrix = adata_fb.to_df()

        main_logger.info(f"modalities to denoise: {features}")

    else:
        raise Exception(file_extension + " files are not supported.")

    if ambient_profile_path:
        _, ambient_profile_file_extension = os.path.splitext(ambient_profile_path)
        if ambient_profile_file_extension == ".pickle":
            ambient_profile = pd.read_pickle(ambient_profile_path)

        # Currently, use the default approach to calculate the ambient profile in the case of h5
        elif ambient_profile_file_extension == ".h5":
            ambient_profile = None

        else:
            raise Exception(
                ambient_profile_file_extension + " files are not supported."
            )
    else:
        ambient_profile = None

    main_logger.info(f"feature_type: {feature_type}")
    main_logger.info(f"count_model: {count_model}")
    main_logger.info(f"output_dir: {output_dir}")
    main_logger.info(f"count_matrix_path: {count_matrix_path}")
    main_logger.info(f"ambient_profile_path: {ambient_profile_path}")
    main_logger.info(f"expected data sparsity: {sparsity:.2f}")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Run model
    scar_model = model(
        raw_count=count_matrix,
        ambient_profile=ambient_profile,
        nn_layer1=nn_layer1,
        nn_layer2=nn_layer2,
        latent_dim=latent_dim,
        feature_type=feature_type,
        count_model=count_model,
        sparsity=sparsity,
        device=device,
    )

    scar_model.train(
        batch_size=batch_size,
        epochs=epochs,
        save_model=save_model,
    )

    scar_model.inference(
        adjust=adjust,
        round_to_int=round_to_int,
        batch_size=batch_size_infer,
        clip_to_obs=clip_to_obs,
    )

    if feature_type.lower() in ["sgrna", "sgrnas", "tag", "tags", "cmo", "cmos"]:
        scar_model.assignment(cutoff=cutoff, moi=moi)

    main_logger.info("Saving results...")

    # save results
    if file_extension == ".pickle":
        output_path01, output_path02, output_path03, output_path04 = (
            os.path.join(output_dir, "denoised_counts.pickle"),
            os.path.join(output_dir, "BayesFactor.pickle"),
            os.path.join(output_dir, "native_frequency.pickle"),
            os.path.join(output_dir, "noise_ratio.pickle"),
        )

        pd.DataFrame(
            scar_model.native_counts,
            index=count_matrix.index,
            columns=count_matrix.columns,
        ).to_pickle(output_path01)
        pd.DataFrame(
            scar_model.bayesfactor,
            index=count_matrix.index,
            columns=count_matrix.columns,
        ).to_pickle(output_path02)
        pd.DataFrame(
            scar_model.native_frequencies,
            index=count_matrix.index,
            columns=count_matrix.columns,
        ).to_pickle(output_path03)
        pd.DataFrame(
            scar_model.noise_ratio, index=count_matrix.index, columns=["noise_ratio"]
        ).to_pickle(output_path04)

        main_logger.info(f"denoised counts saved in: {output_path01}")
        main_logger.info(f"BayesFactor matrix saved in: {output_path02}")
        main_logger.info(f"expected native frequencies saved in: {output_path03}")
        main_logger.info(f"expected noise ratio saved in: {output_path04}")

        if feature_type.lower() in ["sgrna", "sgrnas", "tag", "tags", "cmo", "cmos"]:
            output_path05 = os.path.join(output_dir, "assignment.pickle")
            scar_model.feature_assignment.to_pickle(output_path05)
            main_logger.info(f"assignment saved in: {output_path05}")

    elif file_extension == ".h5":
        output_path_h5ad = os.path.join(
            output_dir, f"filtered_feature_bc_matrix_denoised_{feature_type}.h5ad"
        )

        denoised_adata = adata.copy()
        denoised_adata.X = csr_matrix(scar_model.native_counts)
        denoised_adata.obs["noise_ratio"] = pd.DataFrame(
            scar_model.noise_ratio,
            index=count_matrix.index,
            columns=["noise_ratio"],
        )

        denoised_adata.layers["native_frequencies"] = csr_matrix(
            scar_model.native_frequencies
        )
        denoised_adata.layers["BayesFactor"] = csr_matrix(scar_model.bayesfactor)

        if feature_type.lower() in ["sgrna", "sgrnas", "tag", "tags", "cmo", "cmos"]:
            denoised_adata.obs = denoised_adata.obs.join(scar_model.feature_assignment)

        denoised_adata.write(output_path_h5ad)
        main_logger.info("the denoised h5ad file saved in: {output_path_h5ad}")


class Config:
    """
    The configuration options. Options can be specified as command-line arguments.
    """

    def __init__(self) -> None:
        """Initialize configuration values."""
        self.parser = scar_parser()
        self.namespace = vars(self.parser.parse_args())

    def __getattr__(self, option):
        return self.namespace[option]


def scar_parser():
    """Argument parser"""

    parser = argparse.ArgumentParser(
        description="scAR (single cell Ambient Remover): \
        denoising drop-based single-cell omics data",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s version: {__version__}",
    )
    parser.add_argument(
        "count_matrix",
        type=str,
        nargs="+",
        help="The file of raw count matrix, 2D array (cells x genes) or the path of a filtered_feature_bc_matrix.h5",
    )
    parser.add_argument(
        "-ap",
        "--ambient_profile",
        type=str,
        default=None,
        help="The file of empty profile obtained from empty droplets, 1D array",
    )
    parser.add_argument(
        "-ft",
        "--feature_type",
        type=str,
        default="mRNA",
        help="The feature types, e.g. mRNA, sgRNA, ADT, tag, CMO and ATAC",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None, help="Output directory"
    )
    parser.add_argument(
        "-m", "--count_model", type=str, default="binomial", help="Count model"
    )
    parser.add_argument(
        "-sp",
        "--sparsity",
        type=float,
        default=0.9,
        help="The sparsity of expected native signals",
    )
    parser.add_argument(
        "-hl1",
        "--hidden_layer1",
        type=int,
        default=150,
        help="Number of neurons in the first layer",
    )
    parser.add_argument(
        "-hl2",
        "--hidden_layer2",
        type=int,
        default=100,
        help="Number of neurons in the second layer",
    )
    parser.add_argument(
        "-ls",
        "--latent_dim",
        type=int,
        default=15,
        help="Dimension of latent space",
    )
    parser.add_argument(
        "-epo", "--epochs", type=int, default=800, help="Training epochs"
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="auto",
        help="Device used for training, either 'auto', 'cpu', or 'cuda'",
    )
    parser.add_argument(
        "-s",
        "--save_model",
        type=int,
        default=False,
        help="Save the trained model",
    )
    parser.add_argument(
        "-batchsize",
        "--batchsize",
        type=int,
        default=64,
        help="Batch size for training, set a small value upon out of memory error",
    )
    parser.add_argument(
        "-batchsize_infer",
        "--batchsize_infer",
        type=int,
        default=4096,
        help="Batch size for inference, set a small value upon out of memory error",
    )
    parser.add_argument(
        "-adjust",
        "--adjust",
        type=str,
        default="micro",
        help="""Only used  for calculating Bayesfactors to improve performance,  

                | 'micro' -- adjust the estimated native counts per cell. Default.
                | 'global' -- adjust the estimated native counts globally.
                | False -- no adjustment, use the model-returned native counts.""",
    )
    parser.add_argument(
        "-cutoff",
        "--cutoff",
        type=float,
        default=3,
        help="Cutoff for Bayesfactors. See [Ly2020]_",
    )
    parser.add_argument(
        "-round2int",
        "--round2int",
        type=str,
        default="stochastic_rounding",
        help="Round the counts",
    )

    parser.add_argument(
        "-clip_to_obs",
        "--clip_to_obs",
        type=bool,
        default=False,
        help="clip the predicted native counts by observed counts, \
            use it with caution, as it may lead to overestimation of overall noise.",
    )
    parser.add_argument(
        "-moi",
        "--moi",
        type=float,
        default=None,
        help="Multiplicity of Infection. If assigned, it will allow optimized thresholding, \
        which tests a series of cutoffs to find the best one based on distributions of infections under given moi. \
        See [Dixit2016]_ for details. Under development.",
    )
    parser.add_argument(
        "-verbose",
        "--verbose",
        type=bool,
        default=True,
        help="Whether to print the logging messages",
    )
    return parser


if __name__ == "__main__":
    main()
