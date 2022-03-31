# scAR  

[![scAR](https://img.shields.io/badge/scAR-005AF0?style=for-the-badge&logo=dependabot&logoColor=white.svg)](https://github.com/Novartis/scAR)
![single-cell omics](https://img.shields.io/badge/single_cell_omics-005AF0?style=for-the-badge.svg)
![machine learning](https://img.shields.io/badge/machine_learning-005AF0?style=for-the-badge.svg)
![variational autoencoders](https://img.shields.io/badge/variational_autoencoders-005AF0?style=for-the-badge.svg)
![denoising](https://img.shields.io/badge/denoising-005AF0?style=for-the-badge.svg)

**scAR** (single cell Ambient Remover) is a package for denoising multiple single cell omics data. It can be used for multiple tasks, such as, **sgRNA assignment** for scCRISPRseq, **identity barcode assignment** for cell indexing, **protein denoising** for CITE-seq, **mRNA denoising** for scRNAseq, and etc... It is built using probabilistic deep learning, illustrated as follows:

<img src='docs/img/overview_scAR.png' width="1200">


# Table of Contents

- [Installation](#Installation)
- [Usage](#Usage)
- [Dependencies](#Dependencies)
- [Resources](#Resources)
- [License](#License)
- [Reference](#Reference)

## Installation

Clone this repository,

```sh
$ git clone https://github.com/Novartis/scAR.git
```

Enter the cloned directory:

```sh
$ cd scAR
```

To install the dependencies, create a conda environment:
```sh
$ conda env create -f scAR.yml
```

To activate the scAR conda environment run:
```sh
$ conda activate scAR
```

## Usage

There are two ways to run scAR.

1) Use scAR API if you are Python users

```sh
>>> from scAR import model
>>> scarObj = model(adata.X.to_df(), empty_profile)
>>> scarObj.train()
>>> scarObj.inference()
>>> adata.layers["X_scAR_denoised"] = scarObj.native_counts
>>> adata.obsm["X_scAR_assignment"] = scarObj.feature_assignment  # feature assignment, e.g., sgRNAs, tags, and etc.. Only available in 'cropseq' mode
```

See the [tutorials](#Resources)


2) Run scAR from the command line
```sh
$ scar raw_count_matrix.pickle -t technology -e empty_profile.pickle -o output
```

`raw_count_matrix.pickle`, a pickle-formatted raw count matrix (MxN) with cells in rows and features in columns  
`empty_profile.pickle`, a pickle-formatted feature frequencies (Nx1) in empty droplets  
`technology`, a string, either 'scRNAseq' or 'CROPseq' or 'CITEseq'

Use `scar --help` command to see other optional arguments and parameters.


The output folder contains four (or five) files:   

	output
	├── denoised_counts.pickle		# denoised count matrix
	├── expected_noise_ratio.pickle	# estimated noise ratio
	├── BayesFactor.pickle			# bayesian factor of ambient contamination
	├── expected_native_freq.pickle	# estimated native frequencies
	└── assignment.pickle			# feature assignment, e.g., sgRNAs, tags, and etc.. Gernerated under 'cropseq' mode



## Dependencies

[![PyTorch 1.8](https://img.shields.io/badge/PyTorch-1.8.0-greeen.svg)](https://pytorch.org/)
[![Python 3.8.6](https://img.shields.io/badge/python-3.8.6-blue.svg)](https://www.python.org/)
[![torchvision 0.9.0](https://img.shields.io/badge/torchvision-0.9.0-red.svg)](https://pytorch.org/vision/stable/index.html)
[![torchaudio 0.8.0](https://img.shields.io/badge/torchaudio-0.8.0-yellow.svg)](https://pytorch.org/audio/stable/index.html)
[![tqdm 4.62.3](https://img.shields.io/badge/tqdm-4.62.3-orange.svg)](https://github.com/tqdm/tqdm)
[![scikit-learn 1.0.1](https://img.shields.io/badge/scikit_learn-1.0.1-green.svg)](https://scikit-learn.org/)

## Resources

- Tutorials:
    - [sgRNA Assignment single-cell CRISPR screens](https://github.com/CaibinSh/scAR-reproducibility/blob/main/reproducibility/scAR_tutorial_sgRNA_assignment.ipynb)
    - [Denoising protein data for CITE-seq](https://github.com/CaibinSh/scAR-reproducibility/blob/main/reproducibility/scAR_tutorial_denoising_CITEseq.ipynb)
    - [Denoising mRNA data for scRNAseq](https://github.com/CaibinSh/scAR-reproducibility/blob/main/reproducibility/scAR_tutorial_mRNA_denoising.ipynb)
- If you'd like to contribute, please contact Caibin (caibin.sheng@novartis.com).
- Please use the [issues](https://github.com/Novartis/scAR/issues) to submit bug reports.

## License

This project is licensed under the terms of [License](LICENSE.txt).  
Copyright 2022 Novartis International AG.

## Reference

If you use scAR in your research, please consider citing our [manuscript](https://doi.org/10.1101/2022.01.14.476312),

```
@article {Sheng2022.01.14.476312,
	author = {Sheng, Caibin and Lopes, Rui and Li, Gang and Schuierer, Sven and Waldt, Annick and Cuttat, Rachel and Dimitrieva, Slavica and Kauffmann, Audrey and Durand, Eric and Galli, Giorgio G and Roma, Guglielmo and de Weck, Antoine},
	title = {Probabilistic modeling of ambient noise in single-cell omics data},
	elocation-id = {2022.01.14.476312},
	year = {2022},
	doi = {10.1101/2022.01.14.476312},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2022/01/14/2022.01.14.476312},
	eprint = {https://www.biorxiv.org/content/early/2022/01/14/2022.01.14.476312.full.pdf},
	journal = {bioRxiv}
}
```
