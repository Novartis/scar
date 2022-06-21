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
We validated scAR on scRNAseq for denoising mRNA, scCRISPR-seq for sgRNA assignment, cell multiplexing for tag assignment and CITE-seq for denoising protein counts (ADT). It recovers great number (33% ~ 50%) of cells in scCRISPR-seq and cell multiplexing experiments and significantly improves data quality in scRNAseq and CITE-seq. In theory, any droplet-based single-cell omics technology may have the ambient contamination issue, especially for the complex experiments. scAR can also be a reasonable solution in these cases.