
# Neural Crowd Simulator

This is the code accompanying the paper 
"Discovering interaction mechanisms in crowds via deep generative surrogate experiments" by
Koen Minartz, Fleur Hendriks, Simon Martinus Koop, Alessandro Corbetta, and Vlado Menkovski.

## Setup

To install requirements:

```setup
pip install -r requirements.txt
```

Additionally, install a PyTorch version that is compatible with your hardware, 
and install the corresponding version of torch-scatter.
The code was tested with python 3.9.5, pytorch 1.13.1+cu117 and torch-scatter 2.1.1+pt113cu117.

Finally, download the data
[from this link](https://drive.google.com/drive/folders/1BT_si-IKd_G1aKL3zQj8GeAyn4LtzMgN?usp=sharing)
and unzip it in the root of the repository.

## Running the experiments

To run the experiments, run:

```eval
python evaluation/run_experiments.py
```

The calculations can take some time.

## Model training

To train your own model, run this command, where \<config\> is a config file in the configs directory
(make sure it is imported in configs/\__init\__.py, and do not add the .py extension):

```train
python train_model.py <config>
```

For example:

```train
python train_model.py NeCS_training_default
```


