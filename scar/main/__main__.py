# -*- coding: utf-8 -*-

import argparse, os
import pandas as pd
from ._scar import model
from .__version__ import __version__


def main():

    ##########################################################################################
    ## argument parser
    ##########################################################################################

    parser = argparse.ArgumentParser(
        description="single cell Ambient Remover (scAR): denoising drop-based single-cell omics data",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {version}".format(version=__version__),
    )
    parser.add_argument(
        "count_matrix",
        type=str,
        nargs="+",
        help="the file of observed count matrix, 2D array (cells x genes)",
    )
    parser.add_argument(
        "-e",
        "--empty_profile",
        type=str,
        default=None,
        help="the file of empty profile obtained from empty droplets, 1D array",
    )
    parser.add_argument(
        "-t",
        "--technology",
        type=str,
        default="scRNAseq",
        help="scRNAseq technology, e.g. scRNAseq, CROPseq, CITEseq, ... etc.",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None, help="output directory"
    )
    parser.add_argument(
        "-m", "--count_model", type=str, default="binomial", help="count model"
    )
    parser.add_argument(
        "-tb", "--TensorBoard", type=str, default=False, help="Tensorboard directory"
    )
    parser.add_argument(
        "-hl1",
        "--hidden_layer1",
        type=int,
        default=None,
        help="number of neurons in the first layer",
    )
    parser.add_argument(
        "-hl2",
        "--hidden_layer2",
        type=int,
        default=None,
        help="number of neurons in the second layer",
    )
    parser.add_argument(
        "-ls",
        "--latent_space",
        type=int,
        default=None,
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
        help="whether save the trained model",
    )
    parser.add_argument(
        "-batchsize", "--batchsize", type=int, default=64, help="batch size"
    )
    parser.add_argument(
        "-adjust",
        "--adjust",
        type=str,
        default="micro",
        help="""Only used  for calculating Bayesfactors to improve performance,
                                    'micro' -- adjust the estimated native counts per cell. This can overcome the issue of over- or under-estimation of noise. Default.
                                    'global' -- adjust the estimated native counts globally. This can overcome the issue of over- or under-estimation of noise.
                                    False -- no adjustment, use the model-returned native counts.""",
    )
    parser.add_argument(
        "-ft",
        "--feature_type",
        type=str,
        default="sgRNAs",
        help="Feature types (string), e.g., 'sgRNAs', 'CMOs', 'Tags', and etc..",
    )
    parser.add_argument(
        "-cutoff",
        "--cutoff",
        type=float,
        default=3,
        help="Cutoff for Bayesfactors. Default: 3. See https://doi.org/10.1007/s42113-019-00070-x.",
    )
    parser.add_argument(
        "-MOI",
        "--MOI",
        type=float,
        default=None,
        help="Multiplicity of Infection. If assigned, it will allow optimized thresholding, which tests a series of cutoffs to find the best one based on distributions of infections under given MOI. See http://dx.doi.org/10.1016/j.cell.2016.11.038. Under development.",
    )

    args = parser.parse_args()
    count_matrix_path = args.count_matrix[0]
    empty_profile_path = args.empty_profile
    scRNAseq_tech = args.technology
    output_dir = (
        os.getcwd() if not args.output else args.output
    )  # if None, output to current directory
    count_model = args.count_model
    TensorBoard = args.TensorBoard
    NN_layer1 = args.hidden_layer1
    NN_layer2 = args.hidden_layer2
    latent_space = args.latent_space
    epochs = args.epochs
    save_model = args.save_model
    batch_size = args.batchsize
    adjust = args.adjust
    feature_type = args.feature_type
    cutoff = args.cutoff
    MOI = args.MOI

    count_matrix = pd.read_pickle(count_matrix_path)

    print("===========================================")
    print("scRNAseq_tech: ", scRNAseq_tech)
    print("count_model: ", count_model)
    print("output_dir: ", output_dir)
    print("count_matrix_path: ", count_matrix_path)
    print("empty_profile_path: ", empty_profile_path)
    print("TensorBoard path: ", TensorBoard)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Run model
    scarObj = model(
        raw_count=count_matrix_path,
        empty_profile=empty_profile_path,
        NN_layer1=NN_layer1,
        NN_layer2=NN_layer2,
        latent_space=latent_space,
        scRNAseq_tech=scRNAseq_tech,
        model=count_model,
    )

    scarObj.train(
        batch_size=batch_size,
        epochs=epochs,
        TensorBoard=TensorBoard,
        save_model=save_model,
    )

    scarObj.inference(adjust=adjust)

    if scRNAseq_tech.lower() == "cropseq":
        scarObj.assignment(feature_type=feature_type, cutoff=cutoff, MOI=MOI)

    print("===========================================\n  Saving results...")
    output_path01, output_path02, output_path03, output_path04 = (
        os.path.join(output_dir, "denoised_counts.pickle"),
        os.path.join(output_dir, "BayesFactor.pickle"),
        os.path.join(output_dir, "native_frequency.pickle"),
        os.path.join(output_dir, "noise_ratio.pickle"),
    )

    # save results
    pd.DataFrame(
        scarObj.native_counts, index=count_matrix.index, columns=count_matrix.columns
    ).to_pickle(output_path01)
    pd.DataFrame(
        scarObj.bayesfactor, index=count_matrix.index, columns=count_matrix.columns
    ).to_pickle(output_path02)
    pd.DataFrame(
        scarObj.native_frequencies,
        index=count_matrix.index,
        columns=count_matrix.columns,
    ).to_pickle(output_path03)
    pd.DataFrame(
        scarObj.noise_ratio, index=count_matrix.index, columns=["noise_ratio"]
    ).to_pickle(output_path04)

    print(f"...denoised counts saved in: {output_path01}")
    print(f"...BayesFactor matrix saved in: {output_path02}")
    print(f"...expected native frequencies saved in: {output_path03}")
    print(f"...expected noise ratio saved in: {output_path04}")

    if scRNAseq_tech.lower() == "cropseq":
        output_path05 = os.path.join(output_dir, f"assignment.pickle")
        scarObj.feature_assignment.to_pickle(output_path05)
        print(f"...assignment saved in: {output_path05}")

    print(f"===========================================\n  Done!!!")


if __name__ == "__main__":
    main()