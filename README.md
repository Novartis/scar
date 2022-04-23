# scAR  

[![scAR](https://anaconda.org/bioconda/scar/badges/version.svg)](https://anaconda.org/bioconda/scar)
[![Stars](https://img.shields.io/github/stars/Novartis/scar?logo=GitHub&color=red)](https://github.com/Novartis/scAR)
[![Downloads](https://anaconda.org/bioconda/scar/badges/downloads.svg)](https://anaconda.org/bioconda/scar/files)
[![Documentation Status](https://readthedocs.org/projects/scar-tutorials/badge/?version=latest)](https://scar-tutorials.readthedocs.io/en/latest/?badge=latest)

**scAR** (<u>s</u>ingle-<u>c</u>ell <u>A</u>mbient <u>R</u>emover) is a tool for denoising the ambient signals in droplet-based single cell omics. It can be used for multiple tasks, such as, **sgRNA assignment** for scCRISPRseq, **identity barcode assignment** for cell indexing, **protein denoising** for CITE-seq, **mRNA denoising** for scRNAseq, and etc..

# Table of Contents

- [Installation](#Installation)
- [Dependencies](#Dependencies)
- [Resources](#Resources)
- [License](#License)
- [Reference](#Reference)

## [Installation](https://scar-tutorial.readthedocs.io/en/latest/Installation.html)

## Dependencies

[![PyTorch 1.8](https://img.shields.io/badge/PyTorch-1.8.0-greeen.svg)](https://pytorch.org/)
[![Python 3.8.6](https://img.shields.io/badge/python-3.8.6-blue.svg)](https://www.python.org/)
[![torchvision 0.9.0](https://img.shields.io/badge/torchvision-0.9.0-red.svg)](https://pytorch.org/vision/stable/index.html)
[![tqdm 4.62.3](https://img.shields.io/badge/tqdm-4.62.3-orange.svg)](https://github.com/tqdm/tqdm)
[![scikit-learn 1.0.1](https://img.shields.io/badge/scikit_learn-1.0.1-green.svg)](https://scikit-learn.org/)

## Resources

- Installation, Tutorials, and API can be found in the [documentation](https://scar-tutorial.readthedocs.io/en/latest/index.html).
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
