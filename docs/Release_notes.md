# Changelog

## TODO List
* Multiple ambient profiles
* Plotting functionality
* Reporting functionality
* Transfer learning
* Early stopping

-------------------------------------

<!--next-version-placeholder-->

## v0.4.3 (2022-06-15)
### Fix
* **setup:** Fix a bug to allow sample reasonable numbers of droplets ([`ef6f7e4`](https://github.com/Novartis/scar/commit/ef6f7e4e58fcb1ce8cf463bed3697883f561eba9))
* **main:** Fix a bug in main to set default NN number ([`794ff17`](https://github.com/Novartis/scar/commit/794ff17ac349148aaae24ca9c9927d0179ccd3f4))

### Documentation
* **main:** Add scanpy as dependency ([`252a492`](https://github.com/Novartis/scar/commit/252a492a4d545ed485e9acb208f8e18a25886206))

### Performance
* **main:** Set a separate batchsize_infer parameter for inference ([`8727f04`](https://github.com/Novartis/scar/commit/8727f04da3c934de9d1b14358bee434a972d7849))
* **setup:** Add an option of random sampling droplets to speed up calculation ([`ce042dd`](https://github.com/Novartis/scar/commit/ce042dd120fbe592a089a48b4d584629e63797ca))
* **setup:** Enable manupulate large-scale emptydroplets ([`15f1840`](https://github.com/Novartis/scar/commit/15f18408dcd2ef4bdb1de84b55a136da03fb6244))

## v0.4.2 (2022-06-07)
### Documentation
* Update dependencies ([`784ea63`](https://github.com/Novartis/scar/commit/784ea63a1a55b98592dc69be79d15b3f0c22317c))
* Update dependencies ([`cbf1fc6`](https://github.com/Novartis/scar/commit/cbf1fc6614bd1e559e3b80054f99bd7c05fd3958))
* Change background of logo ([`de267ed`](https://github.com/Novartis/scar/commit/de267ed6546fd9e1aba50594223bbddc57199f56))
* Update readme ([`e97dbf1`](https://github.com/Novartis/scar/commit/e97dbf1f14a9c3fc75fbdbf46c11e22630ddd362))
* Modify scAR_logo ([`1f6e890`](https://github.com/Novartis/scar/commit/1f6e890b662e105e810cda5b4354e0ec3476d8a9))
* Update logo ([`18b51e7`](https://github.com/Novartis/scar/commit/18b51e789d1d2a9bb4a078dff71d93dfb854c640))

### Performance
* Add a setup_anndata method ([#54](https://github.com/Novartis/scar/issues/54)) ([`923b1e5`](https://github.com/Novartis/scar/commit/923b1e5f267f50a6aba765f0c2966080dc375a0f))
* Change sparsity to 1 for scCRISPR-seq and cell indexing ([`d4b2c3d`](https://github.com/Novartis/scar/commit/d4b2c3d4083c9619a205d1c66e361d634ebcb13b))

## v0.4.1 (2022-05-19)
### Feature
* inference: add a round_to_int parameter to round the counts (float) for easy interpretation and better integration into other methods ([#47](https://github.com/Novartis/scar/issues/47)) ([`902a2b9`](https://github.com/Novartis/scar/commit/8694239b1efb4afd24871943e97ad006fab355f8)) ([`8694239`](https://github.com/Novartis/scar/commit/04d30678d29e28ceadd71622c9748edaa7ca8769))

### Build
* setup: replace setup.py with setup.cfg and pyproject.toml ([#51](https://github.com/Novartis/scar/pull/51)) ([`3dc999a`](https://github.com/Novartis/scar/pull/51/commits/c30f4f0270c4a6263bf23c5c3f3619f4436f2890))

### Chore
* unittest: refactor unittest ([#51](https://github.com/Novartis/scar/pull/51)) ([`a597c5f`](https://github.com/Novartis/scar/commit/c34f362697ce88a3604bc8b476b7038165699fe4))
* main: refactor device ([#51](https://github.com/Novartis/scar/pull/51)) ([`d807404`](https://github.com/Novartis/scar/commit/a597c5fd57a79cec921daf2133423ec8a8926019699fe4))

### Documentation
* readthedocs: add scAR_logo image ([#51](https://github.com/Novartis/scar/pull/51)) ([`c34f362`](https://github.com/Novartis/scar/commit/902a2b9cefffd8f883963450712825e939869569))
* tutorials: add ci=None to speed up plotting ([#51](https://github.com/Novartis/scar/pull/51)) ([`902a2b9`](https://github.com/Novartis/scar/commit/3dc999a7d475d08446663bd780d943ba4dffe56c))

## v0.4.0 (2022-05-05)
### Feature
* _scar.py: add a sparsity parameter to control data sparsity ([#44](https://github.com/Novartis/scAR/issues/44)) ([`0c30046`](https://github.com/Novartis/scAR/commit/0c30046aa8d20be88f516b8756789d9fab515b10)) ([`cd33fdd`](https://github.com/Novartis/scAR/commit/cd33fddbd6d7117f459e12b57a936148cde0563f))
* _activation_functions.py: rewrite activation functions ([`f19faa5`](https://github.com/Novartis/scAR/commit/cd33fddbd6d7117f459e12b57a936148cde0563f))

### Documentation
* Modify Changlog.md ([`deb920c`](https://github.com/Novartis/scAR/commit/deb920cdaa3b81a7d6dbccc85231bfa87236cee6))

### Chore
* Pylint: disable R and C message classes ([`9eaaa76`](https://github.com/Novartis/scAR/commit/59707026dc14b6f04ec5e6a8c3a9c992fad3e358))
* Pylint: enable pylint to recognize torch members ([`5970702`](https://github.com/Novartis/scAR/commit/927b4b69bd6f30c23ce4a68d0bf215b35167dd21))
* _scar.py: use plot directive in docstring ([`927b4b6`](https://github.com/Novartis/scAR/commit/f19faa5ecbb782ab292ed246c85d3d2cad3c64fa))
* _data_generator.py: use plot directive in docstring ([`927b4b6`](https://github.com/Novartis/scAR/commit/f19faa5ecbb782ab292ed246c85d3d2cad3c64fa))
---------------------------------------

## v0.3.5 (2022-05-03)
### Documentation
* docs: delete API.rst ([`497b080`](https://github.com/Novartis/scAR/commit/497b080eff15143a34c4d75649ba2e130e1d3006))
* __main__(): update autodoc for the command line interface ([`5ad9986`](https://github.com/Novartis/scAR/commit/5ad998607ec41b91a318ef4bc2c46694ad034dcc))
* tutorials: update colab links in tutorial notebooks ([`5ad9986`](https://github.com/Novartis/scAR/commit/5ad998607ec41b91a318ef4bc2c46694ad034dcc))
* _data_generator.py: add documentations ([`11fa2b8`](https://github.com/Novartis/scAR/commit/11fa2b858ae2162052dd6906d237b16a4f3955de))
* _scar.py: update documentation of Python API ([`11fa2b8`](https://github.com/Novartis/scAR/commit/11fa2b858ae2162052dd6906d237b16a4f3955de))

### Refactor
* _data_generator.py: rename class, uppercase --> lowercase ([`5ad9986`](https://github.com/Novartis/scAR/commit/5ad998607ec41b91a318ef4bc2c46694ad034dcc))

## v0.3.4 (2022-05-01)
### Fix
* setup.py: importing modules of scar in setup.py fails. Change it back to exec(open("scar/main/version.py").read()) ([`3e9d7c3`](https://github.com/Novartis/scAR/commit/74c217bd29af8a137b63fcb5e94f12fe0611be66))
### Style
* docstring: google style --> numpy style ([`cb20b2a`](https://github.com/Novartis/scAR/commit/e89cf54ba8cadc6ffdf8c6249a4752b773351d90))
### Documentation 
* __main__(): autodoc command line interface ([`cb20b2a`](https://github.com/Novartis/scAR/commit/e89cf54ba8cadc6ffdf8c6249a4752b773351d90))


## v0.3.3 (2022-05-01)

### Documentation

* Update documentation ([`b9171a3`](https://github.com/Novartis/scAR/commit/b9171a3015350ac37b0bc44cdb00e4c7aa3c2a67))
* Update documentation ([`44a4409`](https://github.com/Novartis/scAR/commit/44a4409fadf8d124d9b5177cf15f53f00e4524ff))
* Autodoc command line interface ([`0efae6c`](https://github.com/Novartis/scAR/commit/0efae6c26a409553bb8caad5de03c2f38842c139))


## [0.3.2](https://github.com/Novartis/scAR/compare/v0.3.0...v0.3.2) (2022-04-29)


### Feature

* GitHub Action: Add python semantic release ([#36](https://github.com/Novartis/scAR/issues/36)) ([e794242](https://github.com/Novartis/scAR/commit/e79424205022c94b525b10e6cf0672ceb8b63d20))

### Documentation

* Release notes are added to the documentation ([#36](https://github.com/Novartis/scAR/issues/36)) ([e794242](https://github.com/Novartis/scAR/commit/e79424205022c94b525b10e6cf0672ceb8b63d20)), closes [#30](https://github.com/Novartis/scAR/issues/30) [#31](https://github.com/Novartis/scAR/issues/31)



## [0.3.0](https://github.com/Novartis/scAR/compare/v0.2.3...v0.3.0) (2022-04-27)



## [0.2.3](https://github.com/Novartis/scAR/compare/v0.2.2...v0.2.3) (2022-04-19)



## [0.2.2](https://github.com/Novartis/scAR/compare/v0.2.1-beta...v0.2.2) (2022-04-04)


## pre-release

### [0.2.1-beta](https://github.com/Novartis/scAR/compare/v0.2.0-beta...v0.2.1-beta) (2022-04-01)



### [0.2.0-beta](https://github.com/Novartis/scAR/compare/v0.1.6-beta...v0.2.0-beta) (2022-04-01)



### [0.1.6-beta](https://github.com/Novartis/scAR/compare/v0.1.5-beta...v0.1.6-beta) (2022-02-17)



### [0.1.5-beta](https://github.com/Novartis/scAR/compare/v0.1.4-beta...v0.1.5-beta) (2022-02-08)



### [0.1.4-beta](https://github.com/Novartis/scAR/compare/v0.1.3-beta...v0.1.4-beta) (2022-02-03)



### [0.1.3-beta](https://github.com/Novartis/scAR/compare/v0.1.2-beta...v0.1.3-beta) (2022-02-03)



### [0.1.2-beta](https://github.com/Novartis/scAR/compare/v0.1.1-beta...v0.1.2-beta) (2022-02-03)



### [0.1.1-beta](https://github.com/Novartis/scAR/compare/v0.1.0-beta...v0.1.1-beta) (2022-02-03)



### [0.1.0-beta](https://github.com/Novartis/scAR/compare/460add0996eb49ec302fcde0e650f84df4f9e4c0...v0.1.0-beta) (2022-01-30)


### Reverts

* Revert "unknow changes" ([460add0](https://github.com/Novartis/scAR/commit/460add0996eb49ec302fcde0e650f84df4f9e4c0))