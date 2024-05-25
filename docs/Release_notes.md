# CHANGELOG



## v0.6.0 (2024-05-25)

### Feature

* feat(scar): support MPS, but not recommend due to lower speed than CPU on Mac ([`9c3145c`](https://github.com/Novartis/scar/commit/9c3145c11ad03c47157cd4303ad3a50def859cb3))

### Unknown

* Develop (#75)

* fix: refactor pyproject.toml file, delete setup.cfg

* fix: refactor pyproject.toml file, delete setup.cfg

* fix: refactor pyproject.toml file, delete setup.cfg

* chore: update workflow file

* refactor codes (#73) (#74)

* fix: refactor pyproject.toml file, delete setup.cfg

* fix: refactor pyproject.toml file, delete setup.cfg

* fix: refactor pyproject.toml file, delete setup.cfg ([`4d4ea48`](https://github.com/Novartis/scar/commit/4d4ea48fd9cc59e4779c1348e29795feebb69bd3))

* refactor codes (#73)

* fix: refactor pyproject.toml file, delete setup.cfg

* fix: refactor pyproject.toml file, delete setup.cfg

* fix: refactor pyproject.toml file, delete setup.cfg ([`830677b`](https://github.com/Novartis/scar/commit/830677bb0fb733ce63f3aded51d531e1be0bb214))


## v0.5.5 (2024-05-24)

### Unknown

* fix a dependency issue and make a new release ([`401da53`](https://github.com/Novartis/scar/commit/401da532b81bde5c1c4a1f5bc51252c9b9126d34))

* Update pyproject.toml

require setuptools &gt;= 68.1.2 to be compatible with the new version of pytorch-cuda12 ([`d08ff9a`](https://github.com/Novartis/scar/commit/d08ff9a9ec7edde1055ea871525b07b6294fac0c))

* Update __version__.py ([`0af9922`](https://github.com/Novartis/scar/commit/0af992265ef2a2e4f0c6a497d425c42902a72f49))

* Update conf.py ([`9e3fd55`](https://github.com/Novartis/scar/commit/9e3fd55c062940e42e646fcf3b37dabf6f67b314))


## v0.5.4 (2024-05-23)

### Unknown

* Update scar-cpu.yml ([`5c90872`](https://github.com/Novartis/scar/commit/5c90872a09d221428750f6a7c80fd47b72f32ef6))

* Update scar-gpu.yml ([`948381a`](https://github.com/Novartis/scar/commit/948381a53f057e006f6c5aaa968dfde54ba56f03))

* Update setup.cfg ([`98ff44b`](https://github.com/Novartis/scar/commit/98ff44bea7d4d01d04686d9034c232976c1c4270))

* Update index.rst ([`8d80405`](https://github.com/Novartis/scar/commit/8d80405a37ee04712872bf298108499b877d2310))

* Update index.rst ([`df06289`](https://github.com/Novartis/scar/commit/df06289324f1281b05de47c2a1329dd62a4087e6))

* Update Installation.rst ([`e6a7ae5`](https://github.com/Novartis/scar/commit/e6a7ae5a5b0bb8d66148a924666d422875e3f9ca))

* Update requirements.txt ([`2e574d9`](https://github.com/Novartis/scar/commit/2e574d999baa93b358f7d7f2f0c0c042479cf07a))

* Update .readthedocs.yaml

update python version ([`86b948d`](https://github.com/Novartis/scar/commit/86b948dd7148c97d8cfb1192ba87dd61c46db39a))

* Update requirements.txt

remove sphinx ([`bb633db`](https://github.com/Novartis/scar/commit/bb633db73078bdafc829c50c35db1e100961664c))

* Develop (#71)

* fix: update python=3.8.6 -&gt; python=3.10 to fix numpy &amp; pandas conflict

* fix: update the action file

* fix: update the action file

* fix: update the action file

* fix: update the action file

* fix: update the action file

* fix: update the action file

* fix: update version for scanpy and anndata

* fix: version conflict between anndata and scanpy

* fix: remove anndata dependency ([`e8ed18c`](https://github.com/Novartis/scar/commit/e8ed18c1dd5515cd355175ffc39f03a85b06a746))

* Develop (#70)

* fix: update python=3.8.6 -&gt; python=3.10 to fix numpy &amp; pandas conflict

* fix: update the action file

* fix: update the action file

* fix: update the action file

* fix: update the action file

* fix: update the action file

* fix: update the action file

* fix: update version for scanpy and anndata ([`d8d12ed`](https://github.com/Novartis/scar/commit/d8d12ed1f15ff427bff1a5f1dac4b74720177eaa))


## v0.5.3 (2024-05-23)

### Documentation

* docs(release_note): update release notes ([`7e76f32`](https://github.com/Novartis/scar/commit/7e76f320f99618898adb4219535c32c739b21cd4))

### Fix

* fix: update python=3.8.6 -&gt; python=3.10 to fix numpy &amp; pandas conflict (#69)

* fix: update python=3.8.6 -&gt; python=3.10 to fix numpy &amp; pandas conflict

* fix: update the action file

* fix: update the action file

* fix: update the action file

* fix: update the action file

* fix: update the action file

* fix: update the action file ([`ffc4091`](https://github.com/Novartis/scar/commit/ffc40915dbf0921ea89b0cb5ebbd11007feac493))

### Unknown

* Develop (#67)

* revise log info

* update reference

* update reference in introduction ([`14b430b`](https://github.com/Novartis/scar/commit/14b430bf2d3c185ca74d284b87222de57bac9ca6))


## v0.5.2 (2023-05-02)

### Feature

* feat: add a feature/tutorial for denoising scATACseq (#65)

* feat: add a feature/tutorial for denoising scATACseq

* fix(scar): raise MPS NotImplementedError

* chore(scar): add loss in progression bar

* chore(gitignore): ignore count matrix data

* style(main): refactor logging message

* style: refactor __main__.py

* style(_scar): add logging message for device

* chore(_scar.py): clear progressbar

* docs(tutorials): update tutorials

---------

Co-authored-by: EC2 Default User &lt;ec2-user@ip-172-16-91-220.us-east-2.compute.internal&gt; ([`f8432f3`](https://github.com/Novartis/scar/commit/f8432f3aff8e3a1d6489be806ee93dd695029e4c))

### Unknown

* Update __version__.py ([`292a916`](https://github.com/Novartis/scar/commit/292a91643add517e9f2e135f656403626b377957))

* Merge branch &#39;develop&#39; of github.com:Novartis/scar into develop ([`44920dc`](https://github.com/Novartis/scar/commit/44920dc24c9ac28ab7825fe1419784df1515b8ad))


## v0.5.1 (2023-03-02)

### Chore

* chore(release_note): manually add release note ([`54c0c64`](https://github.com/Novartis/scar/commit/54c0c640474de51516536978411c564bbe5747c6))

### Documentation

* docs(tutorial): update hyperlinks for tutorials ([`8db27b6`](https://github.com/Novartis/scar/commit/8db27b632353832ced6489a08835b0cb716035b3))

* docs(tutorials): update tutorials ([`a21e2ec`](https://github.com/Novartis/scar/commit/a21e2ecd507f0c2b9f0e876af04a477927443205))

### Feature

* feat(setup): change sample and n_batch parameters ([`4f5e05f`](https://github.com/Novartis/scar/commit/4f5e05fa5bf6aa413287fb2be73d6ad01a3fc9b3))

* feat: support AnnData object as the input ([`24a1ab6`](https://github.com/Novartis/scar/commit/24a1ab67b64d870f4a785ad8ca13dbb4b45c8293))

* feat: print message ([`7659672`](https://github.com/Novartis/scar/commit/76596727389acc484c3dce78d6f2a262dec33a0e))

* feat: print message ([`203687f`](https://github.com/Novartis/scar/commit/203687fb03f098a1bd2f1f3940d49c7e83421f22))

* feat: estimate ambient_profile_all ([`e1913fd`](https://github.com/Novartis/scar/commit/e1913fd15fdf5546619f750c23e03debbcf1d63e))

### Fix

* fix: fix setupdata ([`63afe9d`](https://github.com/Novartis/scar/commit/63afe9dd8fbc701360715e1a4318f6be8f07e58b))

### Unknown

* Default clip_to_obs to False (#63)

* docs: update documentations

* docs: add a link of binary installers

* perf(main): Command line tool supports a new input: filtered_feature_bc_matrix.h5

* docs(main): update documentation for .h5 file

* docs(main): update dependencies

* docs(main): add link to anndata and scanpy

* docs: update dependencies

* docs: update documentations

* docs: update dependencies

* docs: update dependency

* add myst_parser

* add sphinx_tabs

* autodocsumm==0.2.8

* 0.4.4

Automatically generated by python-semantic-release

* chore: add an option to support Mac M1 and M2 processors

You need to install Mac specific version of Pytorch to allow M1/M2 acceleration. Though, support for this accelerator is still experimental in Pytorch, set device=&#39;cpu&#39; upon any errors.

* chore(docs): correct a wrong link

* chore(docs): fix a typo

* feat(model): add an option to clip the prediction

* chore(docs):revise the docs

* feat(*): remove tensorboard parameter

* remove tensorboard from the command line tool

* change email address, change default training epochs to 400

* 0.5.0

Automatically generated by python-semantic-release

* feat: estimate ambient_profile_all

* fix: fix setupdata

* feat: print message

* feat: print message

* feat: support AnnData object as the input

* default clip_to_obs to False, as it leads to overall over-correction

* default clip_to_obs to False, as it leads to overall over-correction, change docs

* default clip_to_obs to False, as it leads to overall over-correction, change __main__.py

* Update python-semantic-release

* 0.6.0

Automatically generated by python-semantic-release

* make a new release 0.5.1

* feat(setup): change sample and n_batch parameters

* docs(tutorials): update tutorials

* chore(release_note): manually add release note

* docs(tutorial): update hyperlinks for tutorials

---------

Co-authored-by: Sheng, Caibin &lt;caibin.sheng@novartis.com&gt;
Co-authored-by: github-actions &lt;action@github.com&gt;
Co-authored-by: Caibin Sheng &lt;caibin@Caibins-MBP.fritz.box&gt;
Co-authored-by: github-actions &lt;github-actions@github.com&gt; ([`3e66569`](https://github.com/Novartis/scar/commit/3e665696fca90371c5571580921855503805bab5))

* make a new release 0.5.1 ([`dffbd5d`](https://github.com/Novartis/scar/commit/dffbd5dc5c7f7bb88e6fa4b027c930f6d3a507e8))

* make a new release 0.5.1 ([`6922c56`](https://github.com/Novartis/scar/commit/6922c56ecf66222d8a972e7b085d2767241cf798))

* Update python-semantic-release ([`40e906e`](https://github.com/Novartis/scar/commit/40e906e2789509eaa493658d0775f074d571f04c))

* update python-semantic-release ([`77559cd`](https://github.com/Novartis/scar/commit/77559cd49b3ca618cbd65e7582648729dc378ba1))

* Merge branch &#39;main&#39; into develop ([`af55f8b`](https://github.com/Novartis/scar/commit/af55f8bd8809c3b92294dbda2dcc81bdfb910a42))

* default clip_to_obs to False, as it leads to overall over-correction, change __main__.py ([`a527017`](https://github.com/Novartis/scar/commit/a527017dabdbe5140adba58503f0f2a66ee5d5a3))

* default clip_to_obs to False, as it leads to overall over-correction, change docs ([`92c433b`](https://github.com/Novartis/scar/commit/92c433b812004e1ad5602c98405c35f7f071d901))

* default clip_to_obs to False, as it leads to overall over-correction ([`944f09a`](https://github.com/Novartis/scar/commit/944f09aef2a1d9b97200fa90efb4a19eacaa7160))

* new release (#61)

* docs: update documentations

* docs: add a link of binary installers

* perf(main): Command line tool supports a new input: filtered_feature_bc_matrix.h5

* docs(main): update documentation for .h5 file

* docs(main): update dependencies

* docs(main): add link to anndata and scanpy

* docs: update dependencies

* docs: update documentations

* docs: update dependencies

* docs: update dependency

* add myst_parser

* add sphinx_tabs

* autodocsumm==0.2.8

* 0.4.4

Automatically generated by python-semantic-release

* chore: add an option to support Mac M1 and M2 processors

You need to install Mac specific version of Pytorch to allow M1/M2 acceleration. Though, support for this accelerator is still experimental in Pytorch, set device=&#39;cpu&#39; upon any errors.

* chore(docs): correct a wrong link

* chore(docs): fix a typo

* feat(model): add an option to clip the prediction

* chore(docs):revise the docs

* feat(*): remove tensorboard parameter

* remove tensorboard from the command line tool

* change email address, change default training epochs to 400

* 0.5.0

Automatically generated by python-semantic-release

---------

Co-authored-by: Sheng, Caibin &lt;caibin.sheng@novartis.com&gt;
Co-authored-by: github-actions &lt;action@github.com&gt;
Co-authored-by: Caibin Sheng &lt;caibin@Caibins-MBP.fritz.box&gt; ([`3f25b4e`](https://github.com/Novartis/scar/commit/3f25b4e28eec318dc37fbc4affc6fdd7355e5110))


## v0.5.0 (2023-02-04)

### Chore

* chore(docs): fix a typo ([`e7e6dd9`](https://github.com/Novartis/scar/commit/e7e6dd93481fae9a9fc0b17f17e6a2edb8c147b3))

* chore(docs): correct a wrong link ([`7c7deda`](https://github.com/Novartis/scar/commit/7c7deda4bef0b8abb9e21827050c45ec35dd35a6))

* chore: add an option to support Mac M1 and M2 processors

You need to install Mac specific version of Pytorch to allow M1/M2 acceleration. Though, support for this accelerator is still experimental in Pytorch, set device=&#39;cpu&#39; upon any errors. ([`38031ab`](https://github.com/Novartis/scar/commit/38031ab4eb67b5fd939eb39a7cca526f2d276342))

### Feature

* feat(*): remove tensorboard parameter ([`98717e7`](https://github.com/Novartis/scar/commit/98717e7c55f4c4184309cb6d3956097fde661046))

* feat(model): add an option to clip the prediction ([`b250ebf`](https://github.com/Novartis/scar/commit/b250ebfcb412d41430ebcb1fe01b0c4f1b415b11))

* feat(main): support .h5 files by the command line tool (#57)

* docs: update documentations

* docs: add a link of binary installers

* perf(main): Command line tool supports a new input: filtered_feature_bc_matrix.h5

* docs(main): update documentation for .h5 file

* docs(main): update dependencies

* docs(main): add link to anndata and scanpy

* docs: update dependencies

* docs: update documentations

* docs: update dependencies

* docs: update dependency

* add myst_parser

* add sphinx_tabs

* autodocsumm==0.2.8

* 0.4.4

Automatically generated by python-semantic-release

Co-authored-by: github-actions &lt;action@github.com&gt; ([`311ed92`](https://github.com/Novartis/scar/commit/311ed92025e40eef55afb1506684fe30873ae282))

### Unknown

* clip the predicted ambient counts (#60)

* docs: update documentations

* docs: add a link of binary installers

* perf(main): Command line tool supports a new input: filtered_feature_bc_matrix.h5

* docs(main): update documentation for .h5 file

* docs(main): update dependencies

* docs(main): add link to anndata and scanpy

* docs: update dependencies

* docs: update documentations

* docs: update dependencies

* docs: update dependency

* add myst_parser

* add sphinx_tabs

* autodocsumm==0.2.8

* 0.4.4

Automatically generated by python-semantic-release

* chore: add an option to support Mac M1 and M2 processors

You need to install Mac specific version of Pytorch to allow M1/M2 acceleration. Though, support for this accelerator is still experimental in Pytorch, set device=&#39;cpu&#39; upon any errors.

* chore(docs): correct a wrong link

* chore(docs): fix a typo

* feat(model): add an option to clip the prediction

* chore(docs):revise the docs

* feat(*): remove tensorboard parameter

* remove tensorboard from the command line tool

* change email address, change default training epochs to 400

---------

Co-authored-by: Sheng, Caibin &lt;caibin.sheng@novartis.com&gt;
Co-authored-by: github-actions &lt;action@github.com&gt;
Co-authored-by: Caibin Sheng &lt;caibin@Caibins-MBP.fritz.box&gt; ([`76c3e6e`](https://github.com/Novartis/scar/commit/76c3e6e9dd8ced098948b1c74dc7e1aaaa0b7fec))

* Merge branch &#39;main&#39; into develop ([`b524e44`](https://github.com/Novartis/scar/commit/b524e44caccd392405d455fd332fbfaf1ec15d14))

* change email address, change default training epochs to 400 ([`dabd5a2`](https://github.com/Novartis/scar/commit/dabd5a28d10363ea6cb628823eaf981c29fa64c7))

* remove tensorboard from the command line tool ([`6f70fdd`](https://github.com/Novartis/scar/commit/6f70fdd898e4605aa0be79de603973a78e7b8162))

* chore(docs):revise the docs ([`5b15106`](https://github.com/Novartis/scar/commit/5b15106d478f5af315a2a533007b88aa8892c0fd))


## v0.4.4 (2022-08-09)

### Chore

* chore: fix a typo ([`6ddcd8f`](https://github.com/Novartis/scar/commit/6ddcd8f5e4149418b0b31a559aadd65e31b20e3a))

* chore: update docs ([`faf454c`](https://github.com/Novartis/scar/commit/faf454ce4fd7d744f720a7b31027cbebdeef4295))

* chore: update docs ([`d7c3dbd`](https://github.com/Novartis/scar/commit/d7c3dbd104cc6fdb425430be8004a8cbbe8b1439))

* chore: add a dependency ([`a396a73`](https://github.com/Novartis/scar/commit/a396a735dd5cb2e89d736c303b6ee1fca4832096))

* chore(data_generator): sort synthetic data ([`ca52a81`](https://github.com/Novartis/scar/commit/ca52a819e3b09c792ebb5bb7316f594d8b616a6b))

* chore(data_generator): set rasterized=True for heatmaps ([`fd7f794`](https://github.com/Novartis/scar/commit/fd7f7949df11b1b8b162352c6b7351523f0f821f))

### Documentation

* docs: update dependency ([`03cf19e`](https://github.com/Novartis/scar/commit/03cf19e2adfcb84714b7845914feaa45f7b0ae83))

* docs: update dependencies ([`9bd7f1c`](https://github.com/Novartis/scar/commit/9bd7f1c164e7f3e1b30af73dcf9f6b8737d26019))

* docs: update documentations ([`418996c`](https://github.com/Novartis/scar/commit/418996c2dcd3796444d7e01b3b2f1c897c0f7b0d))

* docs: update dependencies ([`1bde351`](https://github.com/Novartis/scar/commit/1bde351ad7f027d36de75f3a60f471e7ef75a6bf))

* docs(main): add link to anndata and scanpy ([`8436e05`](https://github.com/Novartis/scar/commit/8436e0514f5d9f9cdd9ed2c3b9ef036f23232b31))

* docs(main): update dependencies ([`984df35`](https://github.com/Novartis/scar/commit/984df3562f1d4f753ca66caf88619d7305089dcd))

* docs(main): update documentation for .h5 file ([`2a309e0`](https://github.com/Novartis/scar/commit/2a309e0d44eddb75fd1ddf5cdf69fe59e3e40280))

* docs: add a link of binary installers ([`2faed3e`](https://github.com/Novartis/scar/commit/2faed3e56b98027e4ecda68d73044e0c29c369a1))

* docs: update documentations ([`e26a6e9`](https://github.com/Novartis/scar/commit/e26a6e9653e2e41ad46f6fa6aff19a01be00d3ea))

* docs: add competing methods ([`8564b2b`](https://github.com/Novartis/scar/commit/8564b2b5164f72b36cf4e930034fb26951891d4f))

* docs(scar): add versionadded directives for parameter sparsity and round_to_int ([`33e35ca`](https://github.com/Novartis/scar/commit/33e35caf24f15cbe76731586db324cfd52b22511))

* docs: update docs ([`a4da539`](https://github.com/Novartis/scar/commit/a4da5393175e8214f871eb64556cdd6e0f6c632f))

* docs: update introduction ([`a036b24`](https://github.com/Novartis/scar/commit/a036b246b7aed1aa3133bc8df306497afb8942af))

* docs: change readthedocs template ([`421e52f`](https://github.com/Novartis/scar/commit/421e52fe2d23015e4288eeff83fedb105832b2a9))

* docs(data_generator): update docs ([`1f8f668`](https://github.com/Novartis/scar/commit/1f8f66888c42e2433ac712935518ee07b1b4fb1a))

* docs(data_generator): re-style docs ([`afef9fb`](https://github.com/Novartis/scar/commit/afef9fbfd61d827b66139cd9bbf4ff374e2f8a70))

* docs(*): re-style docs ([`2d550fa`](https://github.com/Novartis/scar/commit/2d550fafa816070c153d363a200e0783c863b166))

### Performance

* perf(main): Command line tool supports a new input: filtered_feature_bc_matrix.h5 ([`73bc13e`](https://github.com/Novartis/scar/commit/73bc13e2741d97885840c67177cc985c23749e96))

* perf(setup): add an error raise statement ([`f4fb1a8`](https://github.com/Novartis/scar/commit/f4fb1a8fe915e8deb89c05bf319f85684cbbc853))

### Unknown

* autodocsumm==0.2.8 ([`22420ba`](https://github.com/Novartis/scar/commit/22420ba6c9bffc427e161f4502a023788a4c9523))

* add sphinx_tabs ([`4241244`](https://github.com/Novartis/scar/commit/42412443535a894048dab680fa846d383bec9969))

* add myst_parser ([`a0e8897`](https://github.com/Novartis/scar/commit/a0e8897a6d608933cb10927413dad0bd05a2880f))

* Merge branch &#39;develop&#39; into main ([`f4db01e`](https://github.com/Novartis/scar/commit/f4db01e9026481a6e29856f0d957b8f6f70cbd46))

* Merge branch &#39;develop&#39; into main ([`d600009`](https://github.com/Novartis/scar/commit/d600009bb8eea5a192612e8c321a93bb258a8fb1))

* Merge branch &#39;develop&#39; into main ([`0a039a7`](https://github.com/Novartis/scar/commit/0a039a7e34f202dd109ca8514e5882bef98d7a5e))

* Develop (#56)

* perf(main): set a separate batchsize_infer parameter for inference

* docs(main): add scanpy as dependency

* fix(setup): fix a bug to allow sample reasonable numbers of droplets

* chore(_scar): convert non-str index to str

* 0.4.3

Automatically generated by python-semantic-release

Co-authored-by: github-actions &lt;action@github.com&gt; ([`ab5437e`](https://github.com/Novartis/scar/commit/ab5437e8090c0efa941382c2d0d09fa58f4b59b5))


## v0.4.3 (2022-06-15)

### Chore

* chore(_scar): convert non-str index to str ([`aeb3ea2`](https://github.com/Novartis/scar/commit/aeb3ea2a040098eb16b17ee92673ddf471ae92f5))

### Documentation

* docs(main): add scanpy as dependency ([`252a492`](https://github.com/Novartis/scar/commit/252a492a4d545ed485e9acb208f8e18a25886206))

### Fix

* fix(setup): fix a bug to allow sample reasonable numbers of droplets ([`ef6f7e4`](https://github.com/Novartis/scar/commit/ef6f7e4e58fcb1ce8cf463bed3697883f561eba9))

* fix(main): fix a bug in main to set default NN number ([`794ff17`](https://github.com/Novartis/scar/commit/794ff17ac349148aaae24ca9c9927d0179ccd3f4))

### Performance

* perf(main): set a separate batchsize_infer parameter for inference ([`8727f04`](https://github.com/Novartis/scar/commit/8727f04da3c934de9d1b14358bee434a972d7849))

* perf(setup): add an option of random sampling droplets to speed up calculation ([`ce042dd`](https://github.com/Novartis/scar/commit/ce042dd120fbe592a089a48b4d584629e63797ca))

* perf(setup): enable manupulate large-scale emptydroplets ([`15f1840`](https://github.com/Novartis/scar/commit/15f18408dcd2ef4bdb1de84b55a136da03fb6244))

### Unknown

* Merge branch &#39;develop&#39; into main ([`8232ada`](https://github.com/Novartis/scar/commit/8232ada695a4f547f97c72cdb94a92716b2dd7f9))

* Merge branch &#39;develop&#39; into main ([`53335e5`](https://github.com/Novartis/scar/commit/53335e558b8fff1711d31070d639150749abc15e))

* merge develop to main (#55)

* chore(main): change assigned null sgrna to empty string

* perf: add a setup_anndata method (#54)

* perf: add a setup_anndata method

* fix(docs): downgrade protobuf to allow success when building docs

* chore(setup): refactor codes

* chore(docs): add anndata dependency to docs

* chore(docs): fix wrong hyperlinks of notebooks

* build: downgrade python-semantic-release version

* 0.4.2

Automatically generated by python-semantic-release

Co-authored-by: github-actions &lt;action@github.com&gt; ([`4132481`](https://github.com/Novartis/scar/commit/413248128d06983c9896792c9aef3070e04c41a9))


## v0.4.2 (2022-06-07)

### Build

* build: downgrade python-semantic-release version ([`569b011`](https://github.com/Novartis/scar/commit/569b011016307db211fe8c458ba5bdf678988a08))

### Chore

* chore(docs): fix wrong hyperlinks of notebooks ([`f06726a`](https://github.com/Novartis/scar/commit/f06726a7f2801ab8d601abe2cd7dc170d2196087))

* chore(docs): add anndata dependency to docs ([`e5efca8`](https://github.com/Novartis/scar/commit/e5efca8cf33dee68a011933cd26a82492fe95aee))

* chore(main): change assigned null sgrna to empty string ([`e81cfc3`](https://github.com/Novartis/scar/commit/e81cfc370468124d32420cfcc32671b876b9585c))

### Documentation

* docs: update dependencies ([`784ea63`](https://github.com/Novartis/scar/commit/784ea63a1a55b98592dc69be79d15b3f0c22317c))

* docs: update dependencies ([`cbf1fc6`](https://github.com/Novartis/scar/commit/cbf1fc6614bd1e559e3b80054f99bd7c05fd3958))

* docs: change background of logo ([`de267ed`](https://github.com/Novartis/scar/commit/de267ed6546fd9e1aba50594223bbddc57199f56))

* docs: update readme ([`e97dbf1`](https://github.com/Novartis/scar/commit/e97dbf1f14a9c3fc75fbdbf46c11e22630ddd362))

* docs: modify scAR_logo ([`1f6e890`](https://github.com/Novartis/scar/commit/1f6e890b662e105e810cda5b4354e0ec3476d8a9))

* docs: update logo ([`18b51e7`](https://github.com/Novartis/scar/commit/18b51e789d1d2a9bb4a078dff71d93dfb854c640))

### Performance

* perf: add a setup_anndata method (#54)

* perf: add a setup_anndata method

* fix(docs): downgrade protobuf to allow success when building docs

* chore(setup): refactor codes ([`923b1e5`](https://github.com/Novartis/scar/commit/923b1e5f267f50a6aba765f0c2966080dc375a0f))

* perf: change sparsity to 1 for scCRISPR-seq and cell indexing ([`d4b2c3d`](https://github.com/Novartis/scar/commit/d4b2c3d4083c9619a205d1c66e361d634ebcb13b))

### Unknown

* update readme ([`70b77c0`](https://github.com/Novartis/scar/commit/70b77c0497b45bd35762e4cf9d8f72d18d9ddf35))

* rename logo file ([`7afd964`](https://github.com/Novartis/scar/commit/7afd964c016451a92b9d5d784d4406851d2c60c0))

* Merge branch &#39;develop&#39; into main ([`03881cb`](https://github.com/Novartis/scar/commit/03881cb8830a30256e3f7055387d2e909afd48ef))

* Merge branch &#39;develop&#39; into main ([`f68b594`](https://github.com/Novartis/scar/commit/f68b5940fe2aa81b11085b744b85418d6e7871cc))

* Merge branch &#39;main&#39; into develop ([`af8f574`](https://github.com/Novartis/scar/commit/af8f574a0811b9062fecf0635626cd364a9a05db))


## v0.4.1 (2022-05-19)

### Chore

* chore: refactor setup files ([`c30f4f0`](https://github.com/Novartis/scar/commit/c30f4f0270c4a6263bf23c5c3f3619f4436f2890))

### Ci

* ci(inference): round the counts using stochastic rounding (#50)

* Add option for stochastic rounding to integer (#48)

* ci(inference): add round_to_int parameters in several places

Co-authored-by: mdmanurung &lt;10704760+mdmanurung@users.noreply.github.com&gt; ([`bf6e226`](https://github.com/Novartis/scar/commit/bf6e226099c6d7102742946183a292172891d488))

### Documentation

* docs(readme): add bionconda badge

add badge of &#34;install with bioconda&#34; ([`d807404`](https://github.com/Novartis/scar/commit/d807404aaba22f031304fae41f203ef0d9b361d7))

* docs: update Changelog.md ([`8764941`](https://github.com/Novartis/scar/commit/8764941a8a5d2250cc21c642354f78f89fb26ca4))

### Unknown

* merge develop into main (#51)

* Chore(main): refactor device

* Chore(test): modify unittest

* Chore: add scAR logo in documentation

* Chore: introduce ci=None to speed up kneeplot

* chore: refactor setup files

* ci(inference): round the counts using stochastic rounding (#50)

* Add option for stochastic rounding to integer (#48)

* ci(inference): add round_to_int parameters in several places

Co-authored-by: mdmanurung &lt;10704760+mdmanurung@users.noreply.github.com&gt;

* make a new release

Co-authored-by: mdmanurung &lt;10704760+mdmanurung@users.noreply.github.com&gt; ([`ad76c1c`](https://github.com/Novartis/scar/commit/ad76c1c2ccb52e32c4263dae92aa6689964bb3fb))

* make a new release ([`4ae8e14`](https://github.com/Novartis/scar/commit/4ae8e142ad13202932671691c6ac0d2fe20a2633))

* Chore: introduce ci=None to speed up kneeplot ([`3dc999a`](https://github.com/Novartis/scar/commit/3dc999a7d475d08446663bd780d943ba4dffe56c))

* Chore: add scAR logo in documentation ([`902a2b9`](https://github.com/Novartis/scar/commit/902a2b9cefffd8f883963450712825e939869569))

* Chore(test): modify unittest ([`c34f362`](https://github.com/Novartis/scar/commit/c34f362697ce88a3604bc8b476b7038165699fe4))

* Chore(main): refactor device ([`a597c5f`](https://github.com/Novartis/scar/commit/a597c5fd57a79cec921daf2133423ec8a8926019))

* Merge branch &#39;develop&#39; into main ([`e0cd40a`](https://github.com/Novartis/scar/commit/e0cd40a1a4f8c904840b246d4925ccbf730209fd))

* Chore(datasets): delete dataset folder (#45)

* docs: modify Changlog.md

* Chore(datasets): delete datasets

* Chore(github action): update semantic release conditioning

* Chore(contributing): merge branch &#39;main&#39; into develop

* 0.4.0

Automatically generated by python-semantic-release

* Test(test_scar.py): increase epochs to allow convergence

Co-authored-by: github-actions &lt;action@github.com&gt; ([`241aaca`](https://github.com/Novartis/scar/commit/241aacaa4260176b12dccc20c841ef71eca92e81))

* Test(test_scar.py): increase epochs to allow convergence ([`dfe61e6`](https://github.com/Novartis/scar/commit/dfe61e6b152ceb1381a81e9401736f7665effe8b))


## v0.4.0 (2022-05-05)

### Documentation

* docs(contributing): add contributing guidelines

add contributing guidelines ([`b77967b`](https://github.com/Novartis/scar/commit/b77967ba78e1c19cd88f26c4cc04246cefe86dcf))

* docs: modify Changlog.md ([`deb920c`](https://github.com/Novartis/scar/commit/deb920cdaa3b81a7d6dbccc85231bfa87236cee6))

* docs: update documentation of Python API (#42)

* docs: update documentations

update documentation of Python API
add documentation of synthetic data

* docs: update documentations

update Python API
update command line interface
update colab links in tutorial notebooks
rename class in _data_generator.py

* 0.3.5

Automatically generated by python-semantic-release

Co-authored-by: github-actions &lt;action@github.com&gt; ([`39880f1`](https://github.com/Novartis/scar/commit/39880f11fc7f444aed359ac75ca69a0d8c0fb88d))

### Feature

* feat(scar.model): addition of a sparsity parameter (#44)

* disable C and R message classes

* enable pylint to recognize torch members

* use plot directive in docstring

* feat(scar.main): introduce a sparsity parameter

* rewrite custom activation functions

* update data_generator

* update sparsity parameter (#43)

* update activation function and sparsity parameters

* update tutorials

* fix a bug

* update pylint score to 8.5

* fix a bug in unit test

* update command line tool for sparsity parameter

* add black style badge, readthedocs maxdepth: 3 -&gt;1

* add functional test

* fix a bug in functional testing

* manually update Changelog ([`0c30046`](https://github.com/Novartis/scar/commit/0c30046aa8d20be88f516b8756789d9fab515b10))

* feat(scar.main): introduce a sparsity parameter

1, introduce a sparsity parameter to control data sparsity
2, rewrite custom activation functions ([`cd33fdd`](https://github.com/Novartis/scar/commit/cd33fddbd6d7117f459e12b57a936148cde0563f))

### Unknown

* Chore(contributing): merge branch &#39;main&#39; into develop ([`25cc8a5`](https://github.com/Novartis/scar/commit/25cc8a5dcbc68e26778dcfd020ec41c598f81894))

* Update README.md ([`f9ef8d4`](https://github.com/Novartis/scar/commit/f9ef8d4acf136669de7ec709c0925840efd142fe))

* Update CONTRIBUTING.md ([`ae92a67`](https://github.com/Novartis/scar/commit/ae92a678d3192c0294c18d56c03773d4bfeb2b8d))

* Test(Github Action): test semantic release conditioning ([`2fa5661`](https://github.com/Novartis/scar/commit/2fa566183e469f70217610d4bc314d2bed12e68a))

* Test(Github Action): test semantic release conditioning ([`70813e3`](https://github.com/Novartis/scar/commit/70813e3d0e00eae907712eb67c79cf0bc02d3361))

* Test(Github Action): test semantic release conditioning ([`8096278`](https://github.com/Novartis/scar/commit/80962780d42b3615cd19186bef4cb569abc7fe10))

* Test(Github Action): test semantic release conditioning ([`04a754b`](https://github.com/Novartis/scar/commit/04a754b99a558a456a030bb6ecec0357aeef0e2b))

* Chore(Github Action): update semantic release conditioning ([`52aa26f`](https://github.com/Novartis/scar/commit/52aa26f4edb6e3354c77a303fc3b496f2feb3017))

* Chore(Github Action): update semantic release conditioning ([`c290c08`](https://github.com/Novartis/scar/commit/c290c089a231f70210d8ecac3d27518e656b92c8))

* Chore(Github Action): update semantic release conditioning ([`5e81d73`](https://github.com/Novartis/scar/commit/5e81d7387a71b08c763b12c15515ac95a2b22504))

* Chore(Github Action): update semantic release conditioning ([`d06ec3b`](https://github.com/Novartis/scar/commit/d06ec3b1dd8c4dc467879d9efa674c7bf5c55431))

* Chore(github action): update semantic release conditioning ([`207a6b2`](https://github.com/Novartis/scar/commit/207a6b2ff0942e1a896f8a74c31204f34bd284fb))

* Chore(datasets): delete datasets ([`5a90e8c`](https://github.com/Novartis/scar/commit/5a90e8cb07fe65fb4e34cc56593bbaeafdda7f85))

* update Changlog, update semantic release conditional branch to develop ([`e673d22`](https://github.com/Novartis/scar/commit/e673d225e226abc413499cfdd9f82ae2b1d2c203))

* Merge branch &#39;main&#39; into develop after pull request ([`42aeadf`](https://github.com/Novartis/scar/commit/42aeadfb6d50ea6ebd5445b1d7c20d4924535be4))

* manually update Changelog ([`c2fee96`](https://github.com/Novartis/scar/commit/c2fee9684df455a0040198be2b8700f9d072871c))

* fix a bug in functional testing ([`d463328`](https://github.com/Novartis/scar/commit/d46332817ddc3c8b6542b4ff89da9b0b325f0870))

* add functional test ([`270f10d`](https://github.com/Novartis/scar/commit/270f10dbb24299c3d795b55c1f19008346cf85a1))

* add black style badge, readthedocs maxdepth: 3 -&gt;1 ([`0c6d96e`](https://github.com/Novartis/scar/commit/0c6d96e6f3ea7388013ca35e369ec15f80d7718f))

* update sparsity parameter (#43)

* update activation function and sparsity parameters

* update tutorials

* fix a bug

* update pylint score to 8.5

* fix a bug in unit test

* thank Will

* update command line tool for sparsity parameter ([`0f29c2a`](https://github.com/Novartis/scar/commit/0f29c2af9a2b883e349d4f5448c9cebd9154e9a1))

* update data_generator ([`bcee0c3`](https://github.com/Novartis/scar/commit/bcee0c3eefe01f7132c7a0e51b1919593b7b2928))

* use plot directive in docstring ([`f19faa5`](https://github.com/Novartis/scar/commit/f19faa5ecbb782ab292ed246c85d3d2cad3c64fa))

* enable pylint to recognize torch members ([`927b4b6`](https://github.com/Novartis/scar/commit/927b4b69bd6f30c23ce4a68d0bf215b35167dd21))

* disable C and R message classes ([`5970702`](https://github.com/Novartis/scar/commit/59707026dc14b6f04ec5e6a8c3a9c992fad3e358))

* Merge branch &#39;develop&#39; into main

Conflicts:
	docs/usages/index.rst
	docs/usages/processing.rst
	docs/usages/synthetic_dataset.rst
	scar/main/_data_generater.py ([`9eaaa76`](https://github.com/Novartis/scar/commit/9eaaa7618141d5b4fae9ea0a93c4ef9ce46fb7e7))

* Merge branch &#39;develop&#39; of https://github.com/Novartis/scAR into develop ([`d6961f9`](https://github.com/Novartis/scar/commit/d6961f944f59df56e57d9295c1f1735a75a379a3))


## v0.3.5 (2022-05-03)

### Documentation

* docs: update documentation for data_generator

update docstring for modules of data_generator
update sphinx documentation for data_generator ([`7268ede`](https://github.com/Novartis/scar/commit/7268ede578ad8f8042e5b0c97661ce099f078ec5))

* docs: delete API.rst ([`497b080`](https://github.com/Novartis/scar/commit/497b080eff15143a34c4d75649ba2e130e1d3006))

* docs: update documentations

update Python API
update command line interface
update colab links in tutorial notebooks
rename class in _data_generator.py ([`5ad9986`](https://github.com/Novartis/scar/commit/5ad998607ec41b91a318ef4bc2c46694ad034dcc))

* docs: update documentations
update documentation of Python API
add documentation of synthetic data ([`11fa2b8`](https://github.com/Novartis/scar/commit/11fa2b858ae2162052dd6906d237b16a4f3955de))

### Unknown

* Merge branch &#39;main&#39; into develop ([`4890828`](https://github.com/Novartis/scar/commit/4890828447d25e344f679d84a502b121f24f7c16))

* Update __version__.py

bump to 0.3.4 ([`7dacee8`](https://github.com/Novartis/scar/commit/7dacee8e742a8b25757e91c06fff6bcc0f8c76e9))


## v0.3.4 (2022-05-01)

### Documentation

* docs: autodoc command line interface (#40)

* docs: autodoc command line interface

style: docstring from google to numpy style
docs: autodoc command line interface
refactor: command line interface
test: semantic release on pull_request opened
fix a bug for semantic release
bump to version 0.3.3

Automatically generated by python-semantic-release

Co-authored-by: github-actions &lt;action@github.com&gt; ([`e89cf54`](https://github.com/Novartis/scar/commit/e89cf54ba8cadc6ffdf8c6249a4752b773351d90))

### Fix

* fix: a bug in setup (#41) ([`74c217b`](https://github.com/Novartis/scar/commit/74c217bd29af8a137b63fcb5e94f12fe0611be66))

* fix: a bug in setup ([`6a45f03`](https://github.com/Novartis/scar/commit/6a45f03a2bf9618f4cc4bb691d0af75a518fe1f4))

### Unknown

* Merge branch &#39;main&#39; into develop ([`3e9d7c3`](https://github.com/Novartis/scar/commit/3e9d7c367019614baab3dbdb1ab8a1863067a69d))


## v0.3.3 (2022-05-01)

### Documentation

* docs: autodoc command line interface

style: docstring from google to numpy style
docs: autodoc command line interface
refactor: command line interface ([`0efae6c`](https://github.com/Novartis/scar/commit/0efae6c26a409553bb8caad5de03c2f38842c139))

### Test

* test: semantic release on pull_request opened ([`57aba29`](https://github.com/Novartis/scar/commit/57aba29bf345d582bac44c913f92e1fb73d69364))

### Unknown

* fix a bug for semantic release ([`cb20b2a`](https://github.com/Novartis/scar/commit/cb20b2a49352a9e54ff66df8b37df5f642c0e518))

* fix a bug for semantic release ([`ed1b8a5`](https://github.com/Novartis/scar/commit/ed1b8a552bb3ae2ae619ab47e292830a19dd6bcd))

* refactor codes ([`37074e9`](https://github.com/Novartis/scar/commit/37074e9f8789a9766d493c4a6b673cc70148fc01))

* Merge branch &#39;main&#39; into develop ([`cd1807e`](https://github.com/Novartis/scar/commit/cd1807e892075fd21b15c553ccbeaefd0af71d58))

* Update conf.py

update conf.py ([`a627895`](https://github.com/Novartis/scar/commit/a62789536a53b06c95bfa1217f3d3347b4eb4d81))

* Update setup.py

update setup ([`aeb5e27`](https://github.com/Novartis/scar/commit/aeb5e27163f9664d2c6fecdabcf7b9c80f5d4faa))

* Update semantic-release.yaml

change semantic release to be triggered when opening a PR ([`36a083d`](https://github.com/Novartis/scar/commit/36a083d2edeef87ebecc3d9987120e071b5606d5))

* Update setup.cfg

change github action to happen on develop branch ([`a743000`](https://github.com/Novartis/scar/commit/a743000f52d90e19c149eba137c4fab132faf7e8))

* update Release_notes.md ([`b0df98d`](https://github.com/Novartis/scar/commit/b0df98d1b0bec4b9f015422376586a9dd55c9171))

* Merge branch &#39;main&#39; into develop ([`121577d`](https://github.com/Novartis/scar/commit/121577d4ebbc9a7dccaf9a4532c565ab81b9b394))

* update Release_notes.md ([`4f6f245`](https://github.com/Novartis/scar/commit/4f6f245d42b00da1bfff735ccfcee72126486e17))


## v0.3.2 (2022-04-29)

### Fix

* fix(*): semantic release bugs (#36)

* additions of docstring, autodocs, reference (#30)

* conf for autodocs

* add numpy for autodocs

* add pip install in requirements for sphinx autodocs

* feature semantic release (#31)

* feat(*) addition of semantic releasing

* fix(*): addition of semantic releasing

* 0.3.1

Automatically generated by python-semantic-release

* fix: changelog
docs: adding docstring in documentation
docs: adding Release notes in documentation
docs: adding docstring in documentation
test: adding semantic release
refactor: further refactoring codes


* fix semantic release

* style: semantic release.yaml

Co-authored-by: github-actions &lt;action@github.com&gt;

* update setup.cfg

* fix(*): changelog
docs: adding docstring in documentation
docs: adding Release notes in documentation
docs: adding docstring in documentation
test: adding semantic release
refactor: further refactoring codes
fix semantic release

* fix a bug in setup.cfg

* fix(*): changelog
docs: adding docstring in documentation
docs: adding Release notes in documentation
docs: adding docstring in documentation
test: adding semantic release
refactor: further refactoring codes
fix semantic release

* Release_notes.md

Co-authored-by: github-actions &lt;action@github.com&gt; ([`e794242`](https://github.com/Novartis/scar/commit/e79424205022c94b525b10e6cf0672ceb8b63d20))

* fix(*): changelog
docs: adding docstring in documentation
docs: adding Release notes in documentation
docs: adding docstring in documentation
test: adding semantic release
refactor: further refactoring codes
fix semantic release ([`b9171a3`](https://github.com/Novartis/scar/commit/b9171a3015350ac37b0bc44cdb00e4c7aa3c2a67))

* fix(*): changelog
docs: adding docstring in documentation
docs: adding Release notes in documentation
docs: adding docstring in documentation
test: adding semantic release
refactor: further refactoring codes
fix semantic release ([`44a4409`](https://github.com/Novartis/scar/commit/44a4409fadf8d124d9b5177cf15f53f00e4524ff))

### Unknown

* Release_notes.md ([`72c32b0`](https://github.com/Novartis/scar/commit/72c32b0c1a102106defb455429dca5ffad6142f6))

* fix a bug ([`4d81bde`](https://github.com/Novartis/scar/commit/4d81bdec35f8061933ebb2b4bcb4e51c4e1d049b))

* test semantic release (#34)

* additions of docstring, autodocs, reference (#30)

* conf for autodocs

* add numpy for autodocs

* add pip install in requirements for sphinx autodocs

* feature semantic release (#31)

* feat(*) addition of semantic releasing

* fix(*): addition of semantic releasing

* 0.3.1

Automatically generated by python-semantic-release

* fix: changelog
docs: adding docstring in documentation
docs: adding Release notes in documentation
docs: adding docstring in documentation
test: adding semantic release
refactor: further refactoring codes


* fix semantic release

* style: semantic release.yaml

Co-authored-by: github-actions &lt;action@github.com&gt;

* update setup.cfg

* fix(*): changelog
docs: adding docstring in documentation
docs: adding Release notes in documentation
docs: adding docstring in documentation
test: adding semantic release
refactor: further refactoring codes
fix semantic release

* fix a bug in setup.cfg

* fix(*): changelog
docs: adding docstring in documentation
docs: adding Release notes in documentation
docs: adding docstring in documentation
test: adding semantic release
refactor: further refactoring codes
fix semantic release

Co-authored-by: github-actions &lt;action@github.com&gt; ([`f26a35e`](https://github.com/Novartis/scar/commit/f26a35ec06ca89df6c72a617c347fd088241b101))

* Merge branch &#39;develop&#39; of https://github.com/Novartis/scAR into develop ([`9e010f1`](https://github.com/Novartis/scar/commit/9e010f1129f801ffc1c3ad9409310281a4f39f45))

* fix a bug in setup.cfg ([`0afe77b`](https://github.com/Novartis/scar/commit/0afe77b4e52088347f6758fa3b295fb67cd28719))

* bump to version 0.3.2 (#33)

* additions of docstring, autodocs, reference (#30)

* conf for autodocs

* add numpy for autodocs

* add pip install in requirements for sphinx autodocs

* feature semantic release (#31)

* feat(*) addition of semantic releasing

* fix(*): addition of semantic releasing

* 0.3.1

Automatically generated by python-semantic-release

* fix: changelog
docs: adding docstring in documentation
docs: adding Release notes in documentation
docs: adding docstring in documentation
test: adding semantic release
refactor: further refactoring codes


* fix semantic release

* style: semantic release.yaml

Co-authored-by: github-actions &lt;action@github.com&gt;

* update setup.cfg

* fix(*): changelog
docs: adding docstring in documentation
docs: adding Release notes in documentation
docs: adding docstring in documentation
test: adding semantic release
refactor: further refactoring codes
fix semantic release

Co-authored-by: github-actions &lt;action@github.com&gt; ([`34b711c`](https://github.com/Novartis/scar/commit/34b711ca7a19c009b94bb76d9dee4368e47d0325))

* Merge branch &#39;main&#39; into develop ([`f8eab47`](https://github.com/Novartis/scar/commit/f8eab474956855b4ef2d9cf7b495be38ccf0e7ff))

* additions semantic release and docstring (#32)

* additions of docstring, autodocs, reference (#30)

* conf for autodocs

* add numpy for autodocs

* add pip install in requirements for sphinx autodocs

* feature semantic release (#31)

* feat(*) addition of semantic releasing

* fix(*): addition of semantic releasing

* 0.3.1

Automatically generated by python-semantic-release

* fix: changelog
docs: adding docstring in documentation
docs: adding Release notes in documentation
docs: adding docstring in documentation
test: adding semantic release
refactor: further refactoring codes


* fix semantic release

* style: semantic release.yaml

* update setup.cfg

Co-authored-by: github-actions &lt;action@github.com&gt; ([`f1c0ab8`](https://github.com/Novartis/scar/commit/f1c0ab873b0b801b63565abbbed998573d5a27e7))

* update setup.cfg ([`2eed2d9`](https://github.com/Novartis/scar/commit/2eed2d9bbcbd96534da725a00663e79f840a5b29))

* feature semantic release (#31)

* feat(*) addition of semantic releasing

* fix(*): addition of semantic releasing

* 0.3.1

Automatically generated by python-semantic-release

* fix: changelog
docs: adding docstring in documentation
docs: adding Release notes in documentation
docs: adding docstring in documentation
test: adding semantic release
refactor: further refactoring codes


* fix semantic release

* style: semantic release.yaml

Co-authored-by: github-actions &lt;action@github.com&gt; ([`ae49ba2`](https://github.com/Novartis/scar/commit/ae49ba291ee9896a2aa772f6bc4ece6f3bcfb106))

* add pip install in requirements for sphinx autodocs ([`c0201f7`](https://github.com/Novartis/scar/commit/c0201f7e62df4ae23fa597c6672a4dc3ad99c28a))

* add numpy for autodocs ([`c1299eb`](https://github.com/Novartis/scar/commit/c1299eb1ecf5d3201a8d1b554b52f9dfc5883a40))

* conf for autodocs ([`711689d`](https://github.com/Novartis/scar/commit/711689da54413c2d0493559c0fdfe253fb6ff44f))

* additions of docstring, autodocs, reference (#30) ([`d93970a`](https://github.com/Novartis/scar/commit/d93970ae9106de0a30af043522c8ee1fd311f121))

* git revert commit ([`7869cab`](https://github.com/Novartis/scar/commit/7869cabedd53fe1cbd26fd899c761ca97452dbe5))

* merge main into development ([`d190876`](https://github.com/Novartis/scar/commit/d190876b51be01bccb5e62734e687a24268d673c))


## v0.3.0 (2022-04-27)

### Unknown

* remove datsets to CaibinSh/scAR-reproducibility/data ([`070bf45`](https://github.com/Novartis/scar/commit/070bf45e8c72ec0d86a75b64462673c9ed3a3185))

* tutorials bugs fixed ([`cf8dc02`](https://github.com/Novartis/scar/commit/cf8dc02a4352fd2bd68780d85069401fb1c99faf))

* colab bugs fixed ([`d9b31db`](https://github.com/Novartis/scar/commit/d9b31db7ac5725971782f49e0c822822fe1f1709))

* colab bugs fixed ([`01693b6`](https://github.com/Novartis/scar/commit/01693b65870b1dcd1289878de0b840a31f112e74))

* harmonize development and main branches (#29)

* New release (#26)

* Add github action to publish package to PyPi

* Feature GitHub action conda build (#1)

* Add conda github action that builds the package with conda. It is triggered on pull request.

* Add pyro-ppl and pip as conda dependency. Trigger setup automatically via conda yaml. Split steps in conda build github action.

* Less strict in the setup.py dependencies for torch. Small change in the installation instructions in the documentation.

* Feature - Add basic unit tests to cover activation_functions.py (#3)

* adding first unit tests

* Format conda-build YAML file

* Update action name

* Fix path to __version__

* Test commit - Trigger workflow

* Revert dummy change

Co-authored-by: Mathias Rechenmann &lt;58425548+mr-nvs@users.noreply.github.com&gt;

* Feature clean helper functions (#4)

* delete _helper_functions.py

* delete _hyperparams_optimization.py

* delete plot_every_epoch function

* delete import _helper_functions

* delete gpyopt dependency

* edit only README

* Add pylint github action

* Remove pypi github action (#9)

* support CPU, modify README (#10)

* support CPU, modify README

* shorten the import, add model to __init__

* More lenient scar installation specifications (#8)

* Split scAR.yml to scAR-gpu.yaml and scAR-cpu.yaml.
* More lenient installation specifications.
* Bump version to 0.2.0

* fix a typo, reorganise __init__.py

* fix bugs

* add synthetic data for integration test (#13)

* add synthetic data for integration test

* change paths

* change paths

* Remove torchaudio. Bump version to 0.2.2 (#14)

* Remove torchaudio
* Bump version to 0.2.2

* Update readme (#16)

* update README

* Black github action (#17)

Addition of black github action that runs on every push and every pull request. It shows in the stdout all the changes that need to be made (--diff), but returns exit code 0, even if errors are observed.

* Addition of integration test (#18)

* Add integration test as unit test

* update version

* loweercase module name

* lowercase module name

* Update black.yaml

lowercase module name

* Update python-conda-build.yaml

lowercase module name

* lowercase module name

* Update scar-gpu.yml

* Update scar-cpu.yml

* Update __init__.py

lowercase module name

* refactor versioning (#21)

* refactor versioning

* refactor versioning

* update .gitignore

* fix a bug

* fix bugs

* Documentation with readthedocs (#22)

* Update .readthedocs.yaml

* Update .readthedocs.yaml

* Update .readthedocs.yaml

* Update requirements.txt

* Update requirements.txt

* Create README.md

* Create conf.py

* Update requirements.txt

* Update .readthedocs.yaml

* Update .readthedocs.yaml

* Update requirements.txt

* Update conf.py

* Update .readthedocs.yaml

* add documentations

* fix a bug

* update documentations

* update documentations

* update documentations

* update documentations

* update documentations

* update documentations

* update documentations

* update

* update

* fix a bug

* update

* update

* update

* update

* update

* update

* update

* update

* update

* update document

* update README.md

* upadte documentations

* update documentations

* update documentations

* Update .readthedocs.yaml

* Update README.md

* update documentations

* add comments

* add comments

* update comment functionality

* update documentation

* add documentations via readthedocs (#23)

* fix a bug

* update documentation -- denoising mRNAs

* update documentations -- CITEseq tutorials

* update documentations -- identity barcodes &amp; sgRNA assignment

* Update README.md

add Git Action Badge

* Update README.md

* Update README.md

Update Git Action Badge to default branch and default events

* Black_formatting (#24)

* black_formatting

* Pylint checking

* optimize activation function and add docstring

* fix a bug

* fix a bug for activation functions

* update import torch in activation functions

* refactor _scar.py

* refactor _loss_functions.py

* refactor _vae.py

* refactor __main__.py

* refactor _data_generater.py

* refactor__version__.py

* Pylint formatting

* Update scar-cpu.yml

* Update scar-cpu.yml

* Update scar-cpu.yml

* Update scar-cpu.yml

restrict setuptools&lt;=59.5.0

* Update scar-cpu.yml

* further code refactoring (#25)

* black_formatting

* Pylint checking

* optimize activation function and add docstring

* fix a bug

* fix a bug for activation functions

* update import torch in activation functions

* refactor _scar.py

* refactor _loss_functions.py

* refactor _vae.py

* refactor __main__.py

* refactor _data_generater.py

* refactor__version__.py

* Pylint formatting

* Update scar-cpu.yml

* Update scar-cpu.yml

* Update scar-cpu.yml

* Update scar-cpu.yml

restrict setuptools&lt;=59.5.0

* Update scar-cpu.yml

* refactor _scar.py and _vae.py

* code refactoring

* fix a bug

* fix a bug

* fix a bug

* fix a bug

* increase pylint score from 0.5 to 6

* bump version to 0.2.4

* update tutorials

* optimize command line

* bump version to 0.3.0

Co-authored-by: Gypas, Foivos &lt;foivos.gypas@novartis.com&gt;
Co-authored-by: Foivos Gypas &lt;fgypas@users.noreply.github.com&gt;
Co-authored-by: Mathias Rechenmann &lt;58425548+mr-nvs@users.noreply.github.com&gt;

* delete scAR/test

Co-authored-by: Gypas, Foivos &lt;foivos.gypas@novartis.com&gt;
Co-authored-by: Foivos Gypas &lt;fgypas@users.noreply.github.com&gt;
Co-authored-by: Mathias Rechenmann &lt;58425548+mr-nvs@users.noreply.github.com&gt; ([`2720895`](https://github.com/Novartis/scar/commit/2720895300ebe68219291c328f98a72a36403abd))

* delete scAR/test ([`dbf61eb`](https://github.com/Novartis/scar/commit/dbf61eb188158519e02d879a7a2809a0a5a5626a))

* New release (#26)

* Add github action to publish package to PyPi

* Feature GitHub action conda build (#1)

* Add conda github action that builds the package with conda. It is triggered on pull request.

* Add pyro-ppl and pip as conda dependency. Trigger setup automatically via conda yaml. Split steps in conda build github action.

* Less strict in the setup.py dependencies for torch. Small change in the installation instructions in the documentation.

* Feature - Add basic unit tests to cover activation_functions.py (#3)

* adding first unit tests

* Format conda-build YAML file

* Update action name

* Fix path to __version__

* Test commit - Trigger workflow

* Revert dummy change

Co-authored-by: Mathias Rechenmann &lt;58425548+mr-nvs@users.noreply.github.com&gt;

* Feature clean helper functions (#4)

* delete _helper_functions.py

* delete _hyperparams_optimization.py

* delete plot_every_epoch function

* delete import _helper_functions

* delete gpyopt dependency

* edit only README

* Add pylint github action

* Remove pypi github action (#9)

* support CPU, modify README (#10)

* support CPU, modify README

* shorten the import, add model to __init__

* More lenient scar installation specifications (#8)

* Split scAR.yml to scAR-gpu.yaml and scAR-cpu.yaml.
* More lenient installation specifications.
* Bump version to 0.2.0

* fix a typo, reorganise __init__.py

* fix bugs

* add synthetic data for integration test (#13)

* add synthetic data for integration test

* change paths

* change paths

* Remove torchaudio. Bump version to 0.2.2 (#14)

* Remove torchaudio
* Bump version to 0.2.2

* Update readme (#16)

* update README

* Black github action (#17)

Addition of black github action that runs on every push and every pull request. It shows in the stdout all the changes that need to be made (--diff), but returns exit code 0, even if errors are observed.

* Addition of integration test (#18)

* Add integration test as unit test

* update version

* loweercase module name

* lowercase module name

* Update black.yaml

lowercase module name

* Update python-conda-build.yaml

lowercase module name

* lowercase module name

* Update scar-gpu.yml

* Update scar-cpu.yml

* Update __init__.py

lowercase module name

* refactor versioning (#21)

* refactor versioning

* refactor versioning

* update .gitignore

* fix a bug

* fix bugs

* Documentation with readthedocs (#22)

* Update .readthedocs.yaml

* Update .readthedocs.yaml

* Update .readthedocs.yaml

* Update requirements.txt

* Update requirements.txt

* Create README.md

* Create conf.py

* Update requirements.txt

* Update .readthedocs.yaml

* Update .readthedocs.yaml

* Update requirements.txt

* Update conf.py

* Update .readthedocs.yaml

* add documentations

* fix a bug

* update documentations

* update documentations

* update documentations

* update documentations

* update documentations

* update documentations

* update documentations

* update

* update

* fix a bug

* update

* update

* update

* update

* update

* update

* update

* update

* update

* update document

* update README.md

* upadte documentations

* update documentations

* update documentations

* Update .readthedocs.yaml

* Update README.md

* update documentations

* add comments

* add comments

* update comment functionality

* update documentation

* add documentations via readthedocs (#23)

* fix a bug

* update documentation -- denoising mRNAs

* update documentations -- CITEseq tutorials

* update documentations -- identity barcodes &amp; sgRNA assignment

* Update README.md

add Git Action Badge

* Update README.md

* Update README.md

Update Git Action Badge to default branch and default events

* Black_formatting (#24)

* black_formatting

* Pylint checking

* optimize activation function and add docstring

* fix a bug

* fix a bug for activation functions

* update import torch in activation functions

* refactor _scar.py

* refactor _loss_functions.py

* refactor _vae.py

* refactor __main__.py

* refactor _data_generater.py

* refactor__version__.py

* Pylint formatting

* Update scar-cpu.yml

* Update scar-cpu.yml

* Update scar-cpu.yml

* Update scar-cpu.yml

restrict setuptools&lt;=59.5.0

* Update scar-cpu.yml

* further code refactoring (#25)

* black_formatting

* Pylint checking

* optimize activation function and add docstring

* fix a bug

* fix a bug for activation functions

* update import torch in activation functions

* refactor _scar.py

* refactor _loss_functions.py

* refactor _vae.py

* refactor __main__.py

* refactor _data_generater.py

* refactor__version__.py

* Pylint formatting

* Update scar-cpu.yml

* Update scar-cpu.yml

* Update scar-cpu.yml

* Update scar-cpu.yml

restrict setuptools&lt;=59.5.0

* Update scar-cpu.yml

* refactor _scar.py and _vae.py

* code refactoring

* fix a bug

* fix a bug

* fix a bug

* fix a bug

* increase pylint score from 0.5 to 6

* bump version to 0.2.4

* update tutorials

* optimize command line

* bump version to 0.3.0

Co-authored-by: Gypas, Foivos &lt;foivos.gypas@novartis.com&gt;
Co-authored-by: Foivos Gypas &lt;fgypas@users.noreply.github.com&gt;
Co-authored-by: Mathias Rechenmann &lt;58425548+mr-nvs@users.noreply.github.com&gt; ([`7376aab`](https://github.com/Novartis/scar/commit/7376aab8c33068621e6bcc3d4dca16c7982d0d86))

* bump version to 0.3.0 ([`20ab02e`](https://github.com/Novartis/scar/commit/20ab02e1424ac0d6d282508136210d9eb1a8fba3))

* Merge branch &#39;main&#39; into develop

Conflicts:
	.github/workflows/black.yaml
	README.md
	scar/main/__init__.py ([`e020108`](https://github.com/Novartis/scar/commit/e020108365a7dbe40efc4e6bf57494176e5ee0df))

* further code refactoring (#25)

* black_formatting

* Pylint checking

* optimize activation function and add docstring

* fix a bug

* fix a bug for activation functions

* update import torch in activation functions

* refactor _scar.py

* refactor _loss_functions.py

* refactor _vae.py

* refactor __main__.py

* refactor _data_generater.py

* refactor__version__.py

* Pylint formatting

* Update scar-cpu.yml

* Update scar-cpu.yml

* Update scar-cpu.yml

* Update scar-cpu.yml

restrict setuptools&lt;=59.5.0

* Update scar-cpu.yml

* refactor _scar.py and _vae.py

* code refactoring

* fix a bug

* fix a bug

* fix a bug

* fix a bug

* increase pylint score from 0.5 to 6

* bump version to 0.2.4

* update tutorials

* optimize command line ([`4385b22`](https://github.com/Novartis/scar/commit/4385b22f81885decde3fefc335826afda6cebebf))

* Black_formatting (#24)

* black_formatting

* Pylint checking

* optimize activation function and add docstring

* fix a bug

* fix a bug for activation functions

* update import torch in activation functions

* refactor _scar.py

* refactor _loss_functions.py

* refactor _vae.py

* refactor __main__.py

* refactor _data_generater.py

* refactor__version__.py

* Pylint formatting

* Update scar-cpu.yml

* Update scar-cpu.yml

* Update scar-cpu.yml

* Update scar-cpu.yml

restrict setuptools&lt;=59.5.0

* Update scar-cpu.yml ([`d6e8b73`](https://github.com/Novartis/scar/commit/d6e8b732665bd1eebb17cb162f84ec03be655765))

* Update README.md

Update Git Action Badge to default branch and default events ([`47fbb2a`](https://github.com/Novartis/scar/commit/47fbb2aeb46ee42b4a456f4c1b99c1c3336a8c21))

* Update README.md ([`dc127db`](https://github.com/Novartis/scar/commit/dc127dbfaf8c07aaa598aea4e0a9d8d73758d4b4))

* Update README.md

add Git Action Badge ([`5a6680d`](https://github.com/Novartis/scar/commit/5a6680d72cb398999d4e2982a430c66f572a4268))

* add documentations via readthedocs (#23)

* fix a bug

* update documentation -- denoising mRNAs

* update documentations -- CITEseq tutorials

* update documentations -- identity barcodes &amp; sgRNA assignment ([`88ba2e5`](https://github.com/Novartis/scar/commit/88ba2e5399d4f652f8c44f964edcfaa0770ec065))

* Documentation with readthedocs (#22)

* Update .readthedocs.yaml

* Update .readthedocs.yaml

* Update .readthedocs.yaml

* Update requirements.txt

* Update requirements.txt

* Create README.md

* Create conf.py

* Update requirements.txt

* Update .readthedocs.yaml

* Update .readthedocs.yaml

* Update requirements.txt

* Update conf.py

* Update .readthedocs.yaml

* add documentations

* fix a bug

* update documentations

* update documentations

* update documentations

* update documentations

* update documentations

* update documentations

* update documentations

* update

* update

* fix a bug

* update

* update

* update

* update

* update

* update

* update

* update

* update

* update document

* update README.md

* upadte documentations

* update documentations

* update documentations

* Update .readthedocs.yaml

* Update README.md

* update documentations

* add comments

* add comments

* update comment functionality

* update documentation ([`c4832ba`](https://github.com/Novartis/scar/commit/c4832ba1c089003ae744eecc7662375d18e4f5ef))

* fix bugs ([`26dc46d`](https://github.com/Novartis/scar/commit/26dc46d36303fd924ba8d3799e21a7cb1568f0d1))

* fix a bug ([`b1534fc`](https://github.com/Novartis/scar/commit/b1534fc3c834f174b13f22e416cdbd2ccf45ebd2))

* solve conflict ([`1e54316`](https://github.com/Novartis/scar/commit/1e54316779f59945d1fd4579a6d63e03d761f086))

* update .gitignore ([`441a59d`](https://github.com/Novartis/scar/commit/441a59d5aff47f14a59aa44f97934b0e4ca1007b))

* refactor versioning (#21)

* refactor versioning

* refactor versioning ([`1502294`](https://github.com/Novartis/scar/commit/15022941f39ec5cf274c3dba5947e8f51253faf1))

* Update __init__.py

lowercase module name ([`0766f9f`](https://github.com/Novartis/scar/commit/0766f9f58bc4f17cebba8bb1f9b3786f99a6d68d))

* Update scar-cpu.yml ([`ba51efe`](https://github.com/Novartis/scar/commit/ba51efeecfe6a8ba48d235bfb6b7fe00a3f5b4fe))

* Update scar-gpu.yml ([`a77b6bd`](https://github.com/Novartis/scar/commit/a77b6bdef525fbae6b68348e62f93b67cab66fc6))

* Merge branch &#39;lowcase_package_name&#39; of https://github.com/Novartis/scAR into lowcase_package_name ([`fccbc11`](https://github.com/Novartis/scar/commit/fccbc1112afe8bb064191f7eca336a82019f4231))

* lowercase module name ([`7931b98`](https://github.com/Novartis/scar/commit/7931b984160d0a6c1720023f7f8b380caa846022))

* Update python-conda-build.yaml

lowercase module name ([`f982eef`](https://github.com/Novartis/scar/commit/f982eef2a13ccb97751e11b884659216e1ff0e73))

* Update black.yaml

lowercase module name ([`21c4689`](https://github.com/Novartis/scar/commit/21c4689350851a257d3a1f2076fc96cbfc8ccd40))

* lowercase module name ([`054744e`](https://github.com/Novartis/scar/commit/054744e04bd5a3d73f31b145cef5bc0210e8fe15))

* loweercase module name ([`e3b2fa9`](https://github.com/Novartis/scar/commit/e3b2fa96493ffa2f620cdc3250b93d9033844adb))


## v0.2.3 (2022-04-19)

### Unknown

* Develop (#19)

* Add github action to publish package to PyPi

* Feature GitHub action conda build (#1)

* Add conda github action that builds the package with conda. It is triggered on pull request.

* Add pyro-ppl and pip as conda dependency. Trigger setup automatically via conda yaml. Split steps in conda build github action.

* Less strict in the setup.py dependencies for torch. Small change in the installation instructions in the documentation.

* Feature - Add basic unit tests to cover activation_functions.py (#3)

* adding first unit tests

* Format conda-build YAML file

* Update action name

* Fix path to __version__

* Test commit - Trigger workflow

* Revert dummy change

Co-authored-by: Mathias Rechenmann &lt;58425548+mr-nvs@users.noreply.github.com&gt;

* Feature clean helper functions (#4)

* delete _helper_functions.py

* delete _hyperparams_optimization.py

* delete plot_every_epoch function

* delete import _helper_functions

* delete gpyopt dependency

* edit only README

* Add pylint github action

* Remove pypi github action (#9)

* support CPU, modify README (#10)

* support CPU, modify README

* shorten the import, add model to __init__

* More lenient scar installation specifications (#8)

* Split scAR.yml to scAR-gpu.yaml and scAR-cpu.yaml.
* More lenient installation specifications.
* Bump version to 0.2.0

* fix a typo, reorganise __init__.py

* fix bugs

* add synthetic data for integration test (#13)

* add synthetic data for integration test

* change paths

* change paths

* Remove torchaudio. Bump version to 0.2.2 (#14)

* Remove torchaudio
* Bump version to 0.2.2

* Update readme (#16)

* update README

* Black github action (#17)

Addition of black github action that runs on every push and every pull request. It shows in the stdout all the changes that need to be made (--diff), but returns exit code 0, even if errors are observed.

* Addition of integration test (#18)

* Add integration test as unit test

* update version

Co-authored-by: Gypas, Foivos &lt;foivos.gypas@novartis.com&gt;
Co-authored-by: Foivos Gypas &lt;fgypas@users.noreply.github.com&gt;
Co-authored-by: Mathias Rechenmann &lt;58425548+mr-nvs@users.noreply.github.com&gt; ([`45cbeec`](https://github.com/Novartis/scar/commit/45cbeecbb2a133fedde77f98457ccc3ad5869db0))

* Merge branch &#39;main&#39; into develop ([`c26be41`](https://github.com/Novartis/scar/commit/c26be413b9ed3756160748b01ac1a55bbddb3b60))

* update version ([`f6283c3`](https://github.com/Novartis/scar/commit/f6283c3ae57f1357880b47edc3c247041814d17d))

* Addition of integration test (#18)

* Add integration test as unit test ([`ef7bfb4`](https://github.com/Novartis/scar/commit/ef7bfb4b1d17d52c1afd26f7f6f1b59dac283d6d))

* Black github action (#17)

Addition of black github action that runs on every push and every pull request. It shows in the stdout all the changes that need to be made (--diff), but returns exit code 0, even if errors are observed. ([`7a61e7f`](https://github.com/Novartis/scar/commit/7a61e7fe9d2b6cb0d1597f2b74f9b4e105bfb0e2))

* Update readme (#16)

* update README ([`6018a26`](https://github.com/Novartis/scar/commit/6018a268b14d647344cf309fe8dd5cdb3db806c5))


## v0.2.2 (2022-04-05)

### Unknown

* Remove torchaudio, add test data and bump version to 0.2.2 (#15)

* Add synthetic data for integration test
* Remove torchaudio
* Bump version to 0.2.2

Co-authored-by: Sheng, Caibin &lt;caibin.sheng@novartis.com&gt; ([`e432ad1`](https://github.com/Novartis/scar/commit/e432ad1de81d37b22e707a6f7a5793da9b532ed7))

* Merge branch &#39;main&#39; into develop ([`3bef86e`](https://github.com/Novartis/scar/commit/3bef86eb4870ce4e2bd07bc646ae3bb12f21b8a8))

* Remove torchaudio. Bump version to 0.2.2 (#14)

* Remove torchaudio
* Bump version to 0.2.2 ([`e41e343`](https://github.com/Novartis/scar/commit/e41e343ad66b1b497735c1f5526b2a52a1bdae35))

* add synthetic data for integration test (#13)

* add synthetic data for integration test

* change paths

* change paths ([`1805102`](https://github.com/Novartis/scar/commit/180510213a401bad35dd14010b7bd80af3a45142))

* Develop (#12)

* fix a typo in scAR-gpu.yml
* reorganise init.py files ([`0f973b5`](https://github.com/Novartis/scar/commit/0f973b5f8e326e3b33c746476c315b6d73a37dae))

* Merge branch &#39;main&#39; into develop ([`fe5d4ca`](https://github.com/Novartis/scar/commit/fe5d4ca61a036e37a3819d5d3945f56a653b50a9))

* fix bugs ([`fbd5a94`](https://github.com/Novartis/scar/commit/fbd5a942e78d4e891e345742e05720558d1f78d0))

* fix a typo, reorganise __init__.py ([`2880cfa`](https://github.com/Novartis/scar/commit/2880cfaf3c5405a70110f2c3f2e8864a04d51517))

* 0.2.0-release (#11)

* Support for training of the model with CPUs
* Addition of two yaml files for CPU/GPU installation
* Refactor of setup.py and structure of the package
* Addition of tests with pytest
* Addition of lint checks
* Automate build with github actions (install package and run lint checks and pytest)
* Update documentation
* Version 0.2.0

Co-authored-by: Caibin Sheng &lt;43896555+CaibinSh@users.noreply.github.com&gt;
Co-authored-by: Mathias Rechenmann &lt;58425548+mr-nvs@users.noreply.github.com&gt;
Co-authored-by: Sheng, Caibin &lt;caibin.sheng@novartis.com&gt;
Co-authored-by: Ternent, Tobias &lt;tobias.ternent@novartis.com&gt; ([`61b9782`](https://github.com/Novartis/scar/commit/61b9782b39331e842059a93e1050215a6a0aa1c8))

* More lenient scar installation specifications (#8)

* Split scAR.yml to scAR-gpu.yaml and scAR-cpu.yaml.
* More lenient installation specifications.
* Bump version to 0.2.0 ([`a6ee6dd`](https://github.com/Novartis/scar/commit/a6ee6dd3b83741725d7df04789d15bb7522f41c6))

* support CPU, modify README (#10)

* support CPU, modify README

* shorten the import, add model to __init__ ([`97292da`](https://github.com/Novartis/scar/commit/97292daeabf062a23f90ac8cdec5f405a25014e7))

* Remove pypi github action (#9) ([`8bea098`](https://github.com/Novartis/scar/commit/8bea098c39f0d56eef45b1bf8b199c2caf090be3))

* Merge pull request #7 from Novartis/pylint-github-action

Add pylint github action ([`c1331e0`](https://github.com/Novartis/scar/commit/c1331e03953c1825767562a7a6e8868203717837))

* Add pylint github action ([`38fc141`](https://github.com/Novartis/scar/commit/38fc141c0e71c8d569e7a84b62ef7a1b16b3920c))

* edit only README ([`71a3d5b`](https://github.com/Novartis/scar/commit/71a3d5be60c17811e528a1c65bb42073791efc5b))

* Feature clean helper functions (#4)

* delete _helper_functions.py

* delete _hyperparams_optimization.py

* delete plot_every_epoch function

* delete import _helper_functions

* delete gpyopt dependency ([`501de71`](https://github.com/Novartis/scar/commit/501de715e889690ed977517318e6f8fbb6c08048))

* Feature GitHub action conda build (#1)

* Add conda github action that builds the package with conda. It is triggered on pull request.

* Add pyro-ppl and pip as conda dependency. Trigger setup automatically via conda yaml. Split steps in conda build github action.

* Less strict in the setup.py dependencies for torch. Small change in the installation instructions in the documentation.

* Feature - Add basic unit tests to cover activation_functions.py (#3)

* adding first unit tests

* Format conda-build YAML file

* Update action name

* Fix path to __version__

* Test commit - Trigger workflow

* Revert dummy change

Co-authored-by: Mathias Rechenmann &lt;58425548+mr-nvs@users.noreply.github.com&gt; ([`23ee354`](https://github.com/Novartis/scar/commit/23ee3545e79dd9c1d61a7106314ba1ba4b819978))

* Merge pull request #6 from CaibinSh/feature-pypi-github-action

Add github action to publish package to PyPi ([`a22c86f`](https://github.com/Novartis/scar/commit/a22c86fd4574a3dc8c3d9e85d182ee251fa2d422))

* Add github action to publish package to PyPi ([`959ec7f`](https://github.com/Novartis/scar/commit/959ec7fa2acc4e1cce702d6ca86a7559b9e32632))

* fix a misleading annotation for scCRISPRseq ([`547a7b6`](https://github.com/Novartis/scar/commit/547a7b6baae0396db815bdeac9553063ff931afe))

* update README, add single cell CRISPR tutorial ([`ee507ea`](https://github.com/Novartis/scar/commit/ee507ea82db439eadd439d3fd7ba250f4309c971))

* update commond line tool for guide assignment output ([`bb23a07`](https://github.com/Novartis/scar/commit/bb23a07cdc9ede08da7c5fefe42dfbc688dffb0d))

* integrate sgRNA/identity-barcode assignment module ([`464abbc`](https://github.com/Novartis/scar/commit/464abbc389208643ae7618b7961e54254c9939f1))

* update README, add docs for scAR ([`232343d`](https://github.com/Novartis/scar/commit/232343d096beb9b652a3ca58e65b3e74e01a6e72))

* fix a bug for loss function ([`6e5751c`](https://github.com/Novartis/scar/commit/6e5751c884fa0b6a3109e39b6c339f7e534bfd96))

* adding parameter options for command line tool ([`2859ad9`](https://github.com/Novartis/scar/commit/2859ad91ad2d2ff1b5e44365ae200ad636f02c2a))

* add count adjust options for guide assignment ([`0bac46f`](https://github.com/Novartis/scar/commit/0bac46f7cc5b1df1c27fde5d3645c9cfecf09b8c))

* cleanning ([`0c5b19d`](https://github.com/Novartis/scar/commit/0c5b19d461f59179eb5b9dadf413e53a02555968))

* fix a bug ([`4ae5e41`](https://github.com/Novartis/scar/commit/4ae5e41bee42ac32753d210fde15d02d51b9c80b))

* adding ZeroInflatedPoisson as an option of count model ([`8d02b84`](https://github.com/Novartis/scar/commit/8d02b8496015460797fb8da25738a0ece28a4c66))

* optimize guide assignment by adding error term ([`dbae1a6`](https://github.com/Novartis/scar/commit/dbae1a64385898019c4f24a6bb2168cc13c28895))

* Update _vae.py

add an error term ([`82d2236`](https://github.com/Novartis/scar/commit/82d2236f0b37da93b34d25668c0cb003ab6ae437))

* Update _vae.py

add an error term in inference ([`1b93952`](https://github.com/Novartis/scar/commit/1b9395212e5e44fc3141f12d3096f3920ab1238d))

* Update __main__.py

fix a bug ([`03e04ac`](https://github.com/Novartis/scar/commit/03e04ac973b1c17528ad54ccc3831f0d78a3465d))

* fix a bug for commond line tool, add an error term for BayesFactor computation ([`4e73932`](https://github.com/Novartis/scar/commit/4e73932c6b647ce715f54f75d299d9bdfed438a8))

* add # -*- coding: utf-8 -*- to the first lines of relavant *.py files ([`a8547d3`](https://github.com/Novartis/scar/commit/a8547d38b230f19be3af16a7924ec53e273f9bfd))

* update README file ([`5148832`](https://github.com/Novartis/scar/commit/5148832c3cbe43e770525a62ab64f31818a6eccb))

* delete a citeseq data ([`6d16c11`](https://github.com/Novartis/scar/commit/6d16c11ae94ff37faf783e5021ea61a730bde495))

* update README ([`0eb64b3`](https://github.com/Novartis/scar/commit/0eb64b34aaac26d72e50136c580cc6823c49df92))

* change to ReLU for CITEseq ([`1ff6268`](https://github.com/Novartis/scar/commit/1ff6268382816129d7b7a7266b6c61ab0242e529))

* update tutorial ([`a442a79`](https://github.com/Novartis/scar/commit/a442a79886e1e97ef031b36348a388b612d8af01))

* update README ([`81bbe46`](https://github.com/Novartis/scar/commit/81bbe46a4cab1658d5e3af45fb06e438a6a20261))

* Merge branch &#39;main&#39; of https://github.com/CaibinSh/scAR into main ([`634c050`](https://github.com/Novartis/scar/commit/634c0501d6eea7a4efd1a2e993f320c29fc0b00c))

* update readme ([`e22505e`](https://github.com/Novartis/scar/commit/e22505ecaebcb00d91a2dadd1ebd98d7346ca084))

* add a tutorial ([`7fc4bef`](https://github.com/Novartis/scar/commit/7fc4bef439b2d1423efe9ee7282acae932d3356b))

* Update requirements.txt ([`a21ee16`](https://github.com/Novartis/scar/commit/a21ee1684f5082ec1669552cce2d42f9aa97305c))

* Update requirements.txt ([`07007ca`](https://github.com/Novartis/scar/commit/07007ca5d4fef6be409098ffe6430804fc9d9bd3))

* Update requirements.txt ([`7ee76f7`](https://github.com/Novartis/scar/commit/7ee76f77b226fd7f3a80481dd140d48d5220ac9e))

* Update requirements.txt ([`a6e421a`](https://github.com/Novartis/scar/commit/a6e421a4f5f59aa21b68479f84d627095e4d3269))

* update for readthedocs ([`d43ba5c`](https://github.com/Novartis/scar/commit/d43ba5c909b561df3c7bf909d3b4f881047d1429))

* update inference ([`fce1358`](https://github.com/Novartis/scar/commit/fce1358ebbedc1e6c2119063693d72e1eb3b6481))

* rename scar.readthedocs.yaml ([`2ea19c1`](https://github.com/Novartis/scar/commit/2ea19c142b50478d97bdfdc6069bf5608de5d03f))

* add readthedocs config ([`ab3f7bc`](https://github.com/Novartis/scar/commit/ab3f7bc7e54d44c5a06c3d2a611ca07e3e2d53ec))

* move requirements.txt under docs ([`cce0773`](https://github.com/Novartis/scar/commit/cce077341e790f059aa640d7445a6b430811c9be))

* add scanpy and anndata in scAR.yml env file ([`a179c6d`](https://github.com/Novartis/scar/commit/a179c6d731bb55932e6ebe18f6911e16efd4f326))

* update README file ([`b5f3671`](https://github.com/Novartis/scar/commit/b5f3671203f48d5ec0c3ef56b289972e90164245))

* Merge pull request #1 from CaibinSh/dev

Dev ([`447987b`](https://github.com/Novartis/scar/commit/447987b6ad4a96e43bac5e89be0fd72abf2a6877))

* update env yml file ([`28c0ab7`](https://github.com/Novartis/scar/commit/28c0ab7a7fe9d877bc92284bc9015e0ca2964310))

* modify dataloader to improve efficiency ([`c7a356a`](https://github.com/Novartis/scar/commit/c7a356a53c82bce8d9e1e2283c6e3172c861c9d2))

* improve Dataset and DataLoader to allow more efficient memory usage ([`0d62d53`](https://github.com/Novartis/scar/commit/0d62d5387ce9b7f6ab36cae26c46a45ec96308aa))

* add runtime ([`8d6b21e`](https://github.com/Novartis/scar/commit/8d6b21e3e0b85ed618d27298f73fa451cada9d7e))

* updating dependencies ([`fb53f90`](https://github.com/Novartis/scar/commit/fb53f906b4caecd746474970a181089405be752f))

* Revert &#34;unknow changes&#34;

This reverts commit 8ae88f67e01a424b60d8dfe722f107194e93ae77. ([`460add0`](https://github.com/Novartis/scar/commit/460add0996eb49ec302fcde0e650f84df4f9e4c0))

* unknow changes ([`8ae88f6`](https://github.com/Novartis/scar/commit/8ae88f67e01a424b60d8dfe722f107194e93ae77))

* fix bugs ([`85351e6`](https://github.com/Novartis/scar/commit/85351e63bcf4b8a927d532891738f554a07379e6))

* allow both package and command line tool ([`37fd473`](https://github.com/Novartis/scar/commit/37fd473655e318b2a42f3e7b83151e2ad43ba500))

* split off the simulation modules from _data_loader ([`7c1fc0f`](https://github.com/Novartis/scar/commit/7c1fc0f1c892ee36656bdaae725d521ee5794999))

* rename *.py files ([`d574eee`](https://github.com/Novartis/scar/commit/d574eeedfbdcb5bd5a4cdf3ff22ca594f6e56401))

* split out reproducibility to a separate repo ([`ee554c1`](https://github.com/Novartis/scar/commit/ee554c1f631c218c7faf606e0b82e4f1708e6dbd))

* update dependencies ([`dfa59ac`](https://github.com/Novartis/scar/commit/dfa59ac655b1906750de6b2d4b103a3c8aec7f57))

* update Readme ([`60de3cc`](https://github.com/Novartis/scar/commit/60de3cc5b8b8364cf32fcd2e872f46d3749f396f))

* Update README.md ([`9bc33eb`](https://github.com/Novartis/scar/commit/9bc33eb51821581358bec5c0271545631dd3a9be))

* Update README.md ([`5db8071`](https://github.com/Novartis/scar/commit/5db807188390f20ad1660ebae12b3ac8fe77b576))

* modify img: overview_scAR.png ([`6f56471`](https://github.com/Novartis/scar/commit/6f564714469783f42d64cff2643151fb884407c6))

* Update README.md ([`6d2be13`](https://github.com/Novartis/scar/commit/6d2be13e29b7e75457a6ed90060c6ee02d97432b))

* Update README.md ([`4a775ee`](https://github.com/Novartis/scar/commit/4a775ee86ddbc6bf1ce30834d24c2fe374299cbc))

* Update README.md ([`0b1dad3`](https://github.com/Novartis/scar/commit/0b1dad3ebeba3fffbcef48ad89a61553b8a343ca))

* update reproducibility ([`a6de7be`](https://github.com/Novartis/scar/commit/a6de7be722829842184fd2c4bc815da246bf7e00))

* update reproducibility notebooks ([`0e9eac6`](https://github.com/Novartis/scar/commit/0e9eac61ae2151ec7be4ce8c5e852c65d2936d12))

* update Readme file ([`f232e76`](https://github.com/Novartis/scar/commit/f232e76a32fd8843c2ef6cb99e3e880dbf48de94))

* Add files via upload

initial submission ([`4e71a2a`](https://github.com/Novartis/scar/commit/4e71a2a698423090432e04f7e328719adf7c05b7))

* Delete README.md ([`6f7d79f`](https://github.com/Novartis/scar/commit/6f7d79f5707e7a02a5d24ce78c67ee159d23cbdd))

* Update README.md ([`af33a0c`](https://github.com/Novartis/scar/commit/af33a0c2bd434b58af5271fc8db52b55943886ab))

* Create README.md ([`4695ac3`](https://github.com/Novartis/scar/commit/4695ac32f31af63c1b7993d8c72c80a778c6641e))
