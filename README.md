# tempest_adv
a project for the Current approaches in Machine Learning course at Comenius University, 2026

## training
to train a model (excl. mambair), go into the **train** folder, install *requirements.txt*, and run *train_*.py* with the desired hyperparameters and dataset as arguments. running it without the arguments will provide you with the options.

to train mambair, go into the **mamba_train_eval** folder, install *requirements.txt*, and run *train_mambair.py* with the desired hyperparameters and dataset as arguments.
note: you may encounter difficulties with installing the dependencies - mainly with *mamba-ssm* and *causal-conv1d*. for these packages, i recommend installing them from wheels based on your python, pytorch and cuda versions, from their respective github repositories...

mamba-ssm: https://github.com/state-spaces/mamba/releases

causal-conv1d: https://github.com/Dao-AILab/causal-conv1d/releases

## evaluation
to evaluate a model (excl. mambair) go into the **eval** folder, install *requirements.txt* and *requirements-cer.txt* (optionally, if you want to evaluate CER), and run *evaluate.py* with desired models, checkpoints and dataset as arguments. 

to evaluate mambair, go into the **mamba_train_eval** folder, install *requirements.txt*, and run *evaluate_mambair.py* with mambair checkpoint and dataset as arguments.

## dataset and checkpoints
the dataset and checkpoints can be found at OneDrive: https://liveuniba-my.sharepoint.com/:f:/g/personal/tuch1_uniba_sk/IgAuNaF0ZZTuT5E1illi7amLAZvCWUVaeMxOi1RRjvB-IZc?e=O9HNo6
