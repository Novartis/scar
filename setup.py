from setuptools import setup, find_packages
from glob import glob
import os.path

# Set __version__
exec(open('scAR/__init__.py').read())

scripts = []
for s in glob('scAR/**.py'):
    scripts.append(s)

setup(
    name='scAR',
    version=__version__,
    author="Caibin Sheng",
    # scripts=scripts,
    author_email="caibin.sheng@novartis.com",
    description="single cell Ambient Remover (scAR): remove ambient signals for scRNAseq data",
    packages=find_packages(),
    entry_points = {
        'console_scripts': ['scar=scAR.scAR:main'],},
    include_package_data=True,
    url='https://bitbucket.prd.nibr.novartis.net/users/shengca1/repos/obfx-fbdenoiser/browse',
    license='GPL v3',
    # install_requires=[
    #     "pytorch >= 1.8",
    #     "pandas"
    # ],
    classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",],
    zip_safe=False,
)
