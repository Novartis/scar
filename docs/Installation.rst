Installation
================

To use ``scAR``, first install it using conda install or Git+pip.

Conda install
-------------------------------

1, Install `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_

2, Create conda environment::
    
    conda create -n scar

3, Activate conda environment::
    
    conda activate scar
    
4, Install scar::

    conda install -c bioconda scar
    
5, Activate the scar conda environment::

    conda activate scar
    
Git + pip
-------------------------------------------

1, Clone scar repository::

    git clone https://github.com/Novartis/scAR.git
    
2, Enter the cloned directory::

    cd scAR
    
3, Create a conda environment

.. tabs::

   .. tab:: GPU version
      
      .. code-block::
         :caption: Please use ``scar-gpu`` if you have an nvidia graphics card and the corresponging driver installed
            
            conda env create -f scar-gpu.yml

   .. tab:: CPU version
      
      .. code-block:: 
         :caption: Please use ``scar-cpu`` if you don't have a graphics card availalble
            
            conda env create -f scar-cpu.yml
    
4, Activate the scar conda environment::

    conda activate scar




