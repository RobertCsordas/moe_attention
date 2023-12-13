# Codebase for SwitchHead

The official repository for our paper "SwitchHead: Accelerating Transformers with Mixture-of-Experts Attention".

Please note that this repository is a cleaned-up version of the internal research repository we use. In case you encounter any problems with it, please don't hesitate to contact me.

## Installation

This project requires Python 3.10 and PyTorch 2.1.

```bash
pip3 install -r requirements.txt
```

Create a Weights and Biases account and run
```bash
wandb login
```

More information on setting up Weights and Biases can be found on
https://docs.wandb.com/quickstart.

For plotting, LaTeX is required (to avoid Type 3 fonts and to render symbols). Installation is OS specific.

## Usage

The code makes use of Weights and Biases for experiment tracking. In the "sweeps" directory, we provide sweep configurations for all experiments we have performed. The sweeps are officially meant for hyperparameter optimization, but we use them to run multiple configurations of our models.

To reproduce our results, start a sweep for each of the YAML files in the "sweeps" directory. Run wandb agent for each of them in the main directory. This will run all the experiments, and they will be displayed on the W&B dashboard.

### Re-creating plots from the paper

Edit config file "paper/config.json". Enter your project name in the field "wandb_project" (e.g. "username/modules").

Run the script of interest within the "paper" directory. For example:

```bash
cd paper
python3 plot_datasets.py
```

