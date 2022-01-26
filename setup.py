from setuptools import setup, find_packages
from glob import glob
import os.path

# Set __version__
exec(open('scAR/__init__.py').read())

setup(
    name='scAR',
    version=__version__,
    author="Caibin Sheng",
    author_email="caibin.sheng@novartis.com",
    description="single cell Ambient Remover (scAR): remove ambient signals for single-cell omics data",
    packages=find_packages(),
    entry_points = {
        'console_scripts': ['scar=scAR.__main__:main'],},
    include_package_data=True,
    url='https://github.com/CaibinSh/scAR',
    license='MIT',
    install_requires=[
        "torch==1.10.0",
        "pandas>=1.3.4",
        "torchvision>=0.9.0",
        "torchaudio>=0.8.0",
        "tqdm==4.62.3",
        "gpyopt",
        "seaborn>=0.11.2",
        "tensorboard>=2.2.1",
        "scikit-learn>=1.0.1"        
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific :: Artificial Intelligence"],
    zip_safe=False,
)
