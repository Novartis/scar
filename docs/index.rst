.. scAR documentation master file, created by
   sphinx-quickstart on Fri Apr 22 15:48:44 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

scAR documentation
================================

**Version**: |release|

**Useful links**:
`Binary Installers <https://anaconda.org/bioconda/scar/files>`__ |
`Source Repository <https://github.com/Novartis/scar>`__ |
`Issues <https://github.com/Novartis/scar/issues>`__ |
`Contacts <https://scar-tutorials.readthedocs.io/en/latest/Contacts.html>`__


:mod:`scAR` (single-cell Ambient Remover) is an explainable machine learning model for denoising the ambient signals in droplet-based single cell omics. It can be used for multiple tasks, such as, **sgRNA assignment** in scCRISPR-seq, **identity barcode assignment** in cell multiplexing, **protein denoising** in CITE-seq, **mRNA denoising** in scRNAseq, and etc..

It is developed by Oncology Data Science, Novartis Institute for BioMedical Research.


.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Getting started
      :link: Introduction
      :link-type: doc
      :img-background: _static/bgd.png
      :class-card: sd-text-black
      
      New to *scAR*? Check out the getting started guide. It contains an introduction to *scARs'* main concepts. 
      
      +++
      .. button-ref:: Introduction
         :ref-type: doc
         :color: primary
         :shadow:
         :align: center

         What is scAR?

   .. grid-item-card:: Installation
      :link: Installation
      :link-type: doc
      :img-background: _static/bgd.png
      
      Want to install *scAR*? Check out the installation guide. It contains steps to install *scAR*. 
      
      +++
      .. button-ref:: Installation
         :ref-type: doc
         :color: primary
         :shadow:
         :align: center

         How to install scAR?

   .. grid-item-card:: API reference
      :link: usages/index
      :link-type: doc
      :img-background: _static/bgd.png

      The API reference contains detailed descriptions of scAR API.

      +++
      .. button-ref:: usages/index
         :ref-type: doc
         :color: primary
         :shadow:
         :align: center
      
         To the API

   .. grid-item-card:: Tutorials
      :link: tutorials/index
      :link-type: doc
      :img-background: _static/bgd.png

      The tutorials walk you through the applications of scAR. 
 
      +++
      .. button-ref:: tutorials/index
         :ref-type: doc
         :color: primary
         :shadow:
         :align: center

         To Tutorials

|

.. toctree::
   :hidden:
   
   Introduction
   Installation
   usages/index
   tutorials/index
   Release_notes
   Reference
   License
   Contacts