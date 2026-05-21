# Muscles in Action (ICCV 2023)

[![arXiv](https://img.shields.io/badge/arXiv-2212.02978-b31b1b.svg)](https://arxiv.org/abs/2212.02978)

Code and pre-trained models for the Muscles in Action ICCV 2023 paper. 

## Setup
Environment: 

1. Install a new conda environment enbled for pip packages and compatible Python version:
```commandline
conda create -n musclesinaction python=3.8 pip -y
```
2. Activate environment:
```commandline
conda activate musclesinaction
```
3. Install pip packages:
```commandline
pip install -r requirements.txt
```
4. Download the SMPL model from Google Drive and place it in the project root:

https://drive.usercontent.google.com/download?id=1untXhYOLQtpNEy4GTY_0fL_H-k6cTf_r&authuser=0

5. Run script to load SMPL model and weights:
```commandline
bash scripts/setup_smpl.sh
```

## Dataset: 

The dataset can be found at this link: https://musclesinaction.cs.columbia.edu/MIADataset.tar. Download it, and place the folder in the same directory as the top-level musclesinaction folder.



## Training

To train your own model, run the following command below. By default, it pulls from the musclesinaction/configs/train.yaml file. 

```commandline
$ python musclesinaction/train.py
```

The default is to train a pose-to-emg model, defined with 'predemg=True'. To train an emg-to-pose model, simply set it to False. 

The config file also specifies the information for what data the model is being trained on, as well as where checkpoints are saved, etc. Update it for your goals. 


## Inference

Always enable the conda environment before running any command:
```commandline
$ conda activate musclesinaction
```

The 'musclesinaction/inference_commands' folder has many different scripts to evaluate our model and baselines, per exercise and per person, for both in-distribution and out-of-distribution experiments. 

For instance, to evaluate the emg-to-pose model per exercise, in-distribution, with our model, you would run the following command: 

```commandline
$ python musclesinaction/inference_commands/emgtopose/command_id_cond_exercises_transf_emgtopose.py
```

This will open a tmux session per exercise, and prints the error on the test set for that exercise. 
