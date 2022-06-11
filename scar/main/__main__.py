# -*- coding: utf-8 -*-
"""command line of scar"""

import argparse

# from distutils.command.config import config
import os
import pandas as pd
from ._scar import model
from .__version__ import __version__


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
    TensorBoard = args.TensorBoard
    nn_layer1 = args.hidden_layer1
    nn_layer2 = args.hidden_layer2
    latent_dim = args.latent_dim
    epochs = args.epochs
    sparsity = args.sparsity
    save_model = args.save_model
    batch_size = args.batchsize
    batch_size_infer = args.batchsize_infer
    adjust = args.adjust
    cutoff = args.cutoff
    moi = args.moi
    round_to_int = args.round2int
    count_matrix = pd.read_pickle(count_matrix_path)

    print("===========================================")
    print("feature_type: ", feature_type)
    print("count_model: ", count_model)
    print("output_dir: ", output_dir)
    print("count_matrix_path: ", count_matrix_path)
    print("ambient_profile_path: ", ambient_profile_path)
    print("expected data sparsity: ", sparsity)
    print("TensorBoard path: ", TensorBoard)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Run model
    scar_model = model(
        raw_count=count_matrix_path,
        ambient_profile=ambient_profile_path,
        nn_layer1=nn_layer1,
        nn_layer2=nn_layer2,
        latent_dim=latent_dim,
        feature_type=feature_type,
        count_model=count_model,
        sparsity=sparsity,
    )

    scar_model.train(
        batch_size=batch_size,
        epochs=epochs,
        TensorBoard=TensorBoard,
        save_model=save_model,
    )

    scar_model.inference(
        adjust=adjust, round_to_int=round_to_int, batch_size=batch_size_infer
    )

    if feature_type.lower() in ["sgrna", "sgrnas", "tag", "tags"]:
        scar_model.assignment(cutoff=cutoff, moi=moi)

    print("===========================================\n  Saving results...")
    output_path01, output_path02, output_path03, output_path04 = (
        os.path.join(output_dir, "denoised_counts.pickle"),
        os.path.join(output_dir, "BayesFactor.pickle"),
        os.path.join(output_dir, "native_frequency.pickle"),
        os.path.join(output_dir, "noise_ratio.pickle"),
    )

    # save results
    pd.DataFrame(
        scar_model.native_counts, index=count_matrix.index, columns=count_matrix.columns
    ).to_pickle(output_path01)
    pd.DataFrame(
        scar_model.bayesfactor, index=count_matrix.index, columns=count_matrix.columns
    ).to_pickle(output_path02)
    pd.DataFrame(
        scar_model.native_frequencies,
        index=count_matrix.index,
        columns=count_matrix.columns,
    ).to_pickle(output_path03)
    pd.DataFrame(
        scar_model.noise_ratio, index=count_matrix.index, columns=["noise_ratio"]
    ).to_pickle(output_path04)

    print("...denoised counts saved in: ", output_path01)
    print("...BayesFactor matrix saved in: ", output_path02)
    print("...expected native frequencies saved in: ", output_path03)
    print("...expected noise ratio saved in: ", output_path04)

    if feature_type.lower() in ["sgrna", "sgrnas", "tag", "tags"]:
        output_path05 = os.path.join(output_dir, "assignment.pickle")
        scar_model.feature_assignment.to_pickle(output_path05)
        print("...assignment saved in: ", output_path05)

    print("===========================================\n  Done!!!")


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
        help="the file of observed count matrix, 2D array (cells x genes)",
    )
    parser.add_argument(
        "-ap",
        "--ambient_profile",
        type=str,
        default=None,
        help="the file of empty profile obtained from empty droplets, 1D array",
    )
    parser.add_argument(
        "-ft",
        "--feature_type",
        type=str,
        default="mRNA",
        help="the feature types, e.g. mRNA, sgRNA, ADT and tag",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None, help="output directory"
    )
    parser.add_argument(
        "-m", "--count_model", type=str, default="binomial", help="count model"
    )
    parser.add_argument(
        "-sp",
        "--sparsity",
        type=float,
        default=0.9,
        help="the sparsity of expected native signals",
    )
    parser.add_argument(
        "-tb", "--TensorBoard", type=str, default=False, help="Tensorboard directory"
    )
    parser.add_argument(
        "-hl1",
        "--hidden_layer1",
        type=int,
        default=150,
        help="number of neurons in the first layer",
    )
    parser.add_argument(
        "-hl2",
        "--hidden_layer2",
        type=int,
        default=100,
        help="number of neurons in the second layer",
    )
    parser.add_argument(
        "-ls",
        "--latent_dim",
        type=int,
        default=15,
        help="dimension of latent space",
    )
    parser.add_argument(
        "-epo", "--epochs", type=int, default=800, help="training epochs"
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
        default=None,
        help="Batch size for inference, set a small value upon out of memory error",
    )
    parser.add_argument(
        "-adjust",
        "--adjust",
        type=str,
        default="micro",
        help="""Only used  for calculating Bayesfactors to improve performance,
        'micro' -- adjust the estimated native counts per cell. Default.
        'global' -- adjust the estimated native counts globally.
        False -- no adjustment, use the model-returned native counts.""",
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
        "-moi",
        "--moi",
        type=float,
        default=None,
        help="multiplicity of Infection. If assigned, it will allow optimized thresholding, \
        which tests a series of cutoffs to find the best one based on distributions of infections under given moi. \
        See [Dixit2016]_ for details. Under development.",
    )
    return parser


if __name__ == "__main__":
    main()
