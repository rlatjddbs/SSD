# SSD
Training and evaluation of "Stitching Sub-Trajectories with Conditional Diffusion Model for Goal-Conditioned Offline RL" (preprint). 
The [latest](https://github.com/rlatjddbs/DC/tree/latest) contains the main code for training.

## Prerequisites
1. __Download D4RL__
Follow directions from [this link](https://github.com/Farama-Foundation/D4RL)
2. __Download MuJoCo__
    Before installation of mujoco-py, install dependencies.
    ```
    conda install pkg-config
    conda install patchelf
    conda install -c menpo osmesa
    conda install -c conda-forge mesalib glew glfw
    ```
    Then follow directions from [this link](https://github.com/openai/mujoco-py)
3. __Download Fetch offline dataset__
Download the offline dataset [here](https://drive.google.com/file/d/1niq6bK262segc7qZh8m5RRaFNygEXoBR/view) and place it at `./offline_data`.

## Installation
```
conda env create -n ENV_NAME
conda activate ENV_NAME
pip install -r requirements.txt
```

## Train
```
python train.py
```

## Evaluation
```
python eval.py
```

## Acknowledgements
This repository is based on Michael Janner's [diffuser](https://github.com/jannerm/diffuser) repo. 
