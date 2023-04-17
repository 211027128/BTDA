## Basis Transformation Based Distribution Alignment (BTDA)

This repository contains the code of paper 
<font color=#0099ff>
    'BTDA: Basis Transformation Based Distribution Alignment for Imbalanced Semi-Supervised Learning'.
</font>
We will continuously enhance and refine the code. 
In this repository, we offer a demo and detailed instructions reproduce our proposed classification algorithm.

## Prepare beforehand
Before running or modifing the code, you need to:
+ 1. Clone this repo to your machine.
+ 2. Make sure Anaconda or Miniconda is installed.
+ 3. Run `conda env create -f environment.yml` for environment initialization.

## Run the experiments
It is convenient to perform experiment with this repository. For example, if you want to run BTDA algorithm:

+ 1. Modify the config file in `config/btda/btda_cifar10.yaml` as you need to Make sure there is an imb_factor, where imb_factor=1/$\gamma$
+ 2. Run `python btda_remix.py --c config/btda/btda_cifar10.yaml`

When you want to run other comparison algorithms, just modify the corresponding cofig file and run the code.