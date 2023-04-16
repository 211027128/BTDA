## Usage

Before running or modifing the code, you need to:
1. Clone this repo to your machine.
2. Make sure Anaconda or Miniconda is installed.
3. Run `conda env create -f environment.yml` for environment initialization.

### Run the experiments

It is convenient to perform experiment with TorchSSL. For example, if you want to run BTDA algorithm:

1. Modify the config file in `config/btda/btda_cifar10.yaml` as you need to Make sure there is an imb_factor, where imb_factor=1/$\gamma$
2. Run `python btdah_remix.py --c config/btda/btda_cifar10.yaml`