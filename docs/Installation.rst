Installation
===============

To use scAR, first install it using conda install or Git+pip.

Conda install
------------------------

1, Install `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_

2, Create conda environment

.. code-block:: console

   conda create -n scAR
    
3, Activate conda environment

.. code-block:: console
   
   conda activate scAR

4, Install scar

.. code-block:: console
   
   conda install -c bioconda scar


Git + pip
------------
1, Clone scAR repository,

.. code-block:: console
   
   git clone https://github.com/Novartis/scAR.git

2, Enter the cloned directory:

.. code-block:: console
   
   cd scAR

3, Create a conda environment,

.. note::
    Please use `scAR-gpu` if you have an nvidia graphis card and the corresponging driver installed.
    
    .. code-block:: console
       
       conda env create -f scAR-gpu.yml
   
or

.. note::
    Please use `scAR-cpu` if you don't have a graphis card availalble.
    
    .. code-block:: console
       
       conda env create -f scAR-cpu.yml

4, Activate the scAR conda environment,

.. code-block:: console
   
   conda activate scAR