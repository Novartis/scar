Introduction
===============

What is ambient signal?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: _static/ambient_signal_hypothesis.png
   :width: 500
   :align: center

During the preparation of the single-cell solution, RNA or protein counts are released upon cell lysis and consequently encapsulated by droplets. These exogenous molecules are mixed with native ones and barcoded by the same 10x beads, resulting in overestimated count data. The ambient signal harms the downstream analysis and even introduces significant bias in some cases (e.g. scCRISPR-seq and Cell multiplexing).

The design of scAR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: _static/overview_scAR.png
   :width: 600
   :align: center

scAR uses a latent variable model to represent the biological and technical components in the observed count data. scAR is built under ambient signal hypothesis, in which the probability of occurrence of each ambient transcript can be empirically estimated from cell-free droplets. There are two hidden variables, contamination level per cell and the probability of occurrence of native transcript. With these three parameters, we are able to reconstruct the noisy observations. To learn the hidden variables, we train neural networks (the variational autoencoder) to minimize the differences between the reconstructions and original noisy observations. Once converging, the contamination levels and native expression are inferred and downstream analysis can be performed using these values. See our manuscript [Sheng2022]_ for details.

What types of data that scAR can process?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We validated scAR on scRNAseq to remove ambient mRNA, scCRISPR-seq to assign sgRNAs, cell multiplexing to identify the true tags and CITE-seq to clean noisy protein counts (ADT). It recovers a great number (33% ~ 50%) of cells in scCRISPR-seq and cell multiplexing experiments and significantly improves data quality in scRNAseq and CITE-seq. In theory, any droplet-based single-cell omics technology should have the ambient contamination issue, especially for the complex experiments or samples. scAR can be a reasonable solution in these cases.

What are the alternative apporaches?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
There are several methods to model the data noise in single-cell omics. In general, they can be categorized into two classes. One is dealing with background noise and the other is modeling the stachastic noise. Some of them are listed below.

+-------------------------------------------+-------------------------------------------+
| Background noise                          | Stachastic noise                          |
+========+===============+==================+========+===============+==================+
| CellBender [Fleming2019]_                 | scVI [Lopez2018]_                         |
+-------------------------------------------+-------------------------------------------+
| SoupX [Young2020]_                        | DCA [Eraslan2019]_                        |
+-------------------------------------------+-------------------------------------------+
| DecontX [Yang2020]_                       |                                           |
+-------------------------------------------+-------------------------------------------+
| totalVI (protein counts) [Gayoso2021]_    |                                           |
+-------------------------------------------+-------------------------------------------+
| DSB  (protein counts) [Mulè2022]_         |                                           |
+-------------------------------------------+-------------------------------------------+