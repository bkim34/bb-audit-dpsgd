# Black-box Auditing of DP-SGD
This repository contains the source code for the paper _Nearly Tight Black-Box Auditing of Differentially Private Machine Learning_ by M.S.M.S. Annamalai, E. De Cristofaro, to appear at [NeurIPS 2024](https://arxiv.org/pdf/2405.14106).

## Install
Dependencies are managed by `conda/mamba`.  
1. Required dependencies can be installed using the command `conda env create -f env.yml` and then run `conda activate bb_audit_dpsgd`.  
2. The pre-training algorithm to craft worst-case initial model parameters are given in `craft_inital_params.ipynb`, but we also provide the pre-trained worst-case inital parameters we use under the `pretrained_models/` folder.

## Preparing datasets
Here, we provide the splits we use for the MNIST and CIFAR-10 dataset.
We also provide the last layer activations extracted from running CIFAR-10 inference on the WRN-28-10 model pre-trained on the ImageNet-32 dataset (see [paper](https://arxiv.org/pdf/2204.13650) and [model](https://console.cloud.google.com/storage/browser/dm_jax_privacy/models?authuser=0&pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22)))).
1. In order to de-compress the datasets, use command `cat data_compressed/* | tar -xvz`.

## Run
To audit a Logistic Regression model trained on the MNIST dataset with $\varepsilon = 10.0$ you can run the following command:  
(More command line options can be found inside the `audit_model.py` file)
```bash
$ python3 audit_model.py --data_name mnist --model_name lr --epsilon 10.0
```

## Experiments
We provide the exact scripts we use to run experiments under the `scripts/` folder, which should have more options that you can play around with.  
Results can be plotted using `plot_results.ipynb` notebook.

## Notes
1. We only consider full batch gradient descent, so $B = |D|$ always.
2. For DP-SGD, we sum the gradients instead of averaging them as the size of the dataset can leak information in add/remove DP (see [issue](https://github.com/pytorch/opacus/issues/571)). Therefore, learning rates are expressed in a non-standard way (as $\frac{\eta}{B}$ instead of just $\eta$) here. Specifically, when training a model on half of the MNIST dataset, $\eta = 4$ corresponds to a learning rate of $\frac{\eta}{B} = \frac{4}{30,000} = 1.33 \times 10^{-4}$ (see `scripts/model_init.sh`), which stays the same regardless of whether the dataset is $D$ or $D^-$.