# Muscles in Action (ICCV 2023)

[![arXiv](https://img.shields.io/badge/arXiv-2212.02978-b31b1b.svg)](https://arxiv.org/abs/2212.02978)

Code and pre-trained models for the Muscles in Action ICCV 2023 paper. 

## Setup
Environment Prerequisites: 

0. Install ImageMagick to enable matplotlib animation support:
```commandline
sudo apt-get update && sudo apt-get install -y imagemagick
```

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

4. Download the official dataset from the link below and place it in the project root:

https://musclesinaction.cs.columbia.edu/MIADataset.tar

5. Run script to load dataset:
```commandline
bash scripts/setup_dataset.sh
```

6. Download the SMPL model from Google Drive and place it in the project root:

https://drive.usercontent.google.com/download?id=1untXhYOLQtpNEy4GTY_0fL_H-k6cTf_r&authuser=0

7. Run script to load SMPL model and weights:
```commandline
bash scripts/setup_smpl.sh
```

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
conda activate musclesinaction
```

The `musclesinaction/inference_commands` folder contains orchestrator scripts to run and evaluate our models and baselines. Inference commands are organized into two subdirectories based on prediction direction:
- `emgtopose/`: Scripts to predict 3D/2D Pose from Muscle Activity (sEMG).
- `posetoemg/`: Scripts to predict Muscle Activity (sEMG) from Pose.

For instance, to evaluate the emg-to-pose model per exercise, in-distribution, with our Transformer model, you would run:
```commandline
python musclesinaction/inference_commands/emgtopose/command_id_cond_exercises_transf_emgtopose.py
```
This will open a tmux session per exercise and print the test set error for that exercise.

To attach to a tmux session and view the live execution, use the `-t` (target) flag:
```commandline
tmux attach-session -t <session_name>
```
For example, if the session is named `my_session_clean_Running`, run:
```commandline
tmux attach-session -t my_session_clean_Running
```
You can detach from the session and leave it running in the background by pressing `Ctrl-b` followed by `d`.

To clean the entire tmux context (kill all running sessions), run:
```commandline
tmux kill-server
```
Alternatively, to kill a specific session, you can run:
```commandline
tmux kill-session -t <session_name>
```

### **Inference Script Categorization & Dimensions**

When running or selecting inference scripts, files are named and organized according to four key experimental dimensions:

#### **1. Model Approach: `transf` vs. `retrieval`**
- **`transf`**: Uses a **Transformer** neural network to predict the target modality (e.g. pose from muscle signals, or vice versa).
- **`retrieval`**: Uses a **Nearest Neighbor** search baseline to retrieve the closest matching sample from the training database.

#### **2. Generalization Type: `id` vs. `ood`**
- **`id` (In-Distribution)**: Tests the model on subjects or exercises it has seen during training, but on unseen time segments.
- **`ood` (Out-of-Distribution)**: Tests the model's zero-shot generalization capabilities on **entirely new** people (subject-to-subject) or exercises that were excluded from the training set.

#### **3. Subject/Action Focus: `people` vs. `exercises`**
- **`people`**: Focuses on Subject-to-Subject generalization (evaluating performance when testing on unseen individuals).
- **`exercises`**: Focuses on Exercise-to-Exercise generalization (evaluating performance when testing on unseen movements).

#### **4. Conditioning: `cond` vs. `nocond`**
- **`cond` (Conditional)**: The model is explicitly conditioned on the subject identity or exercise category being performed (e.g. knowing it is a squat).
- **`nocond` (Non-conditional)**: The model is given inputs blindly and must predict the outputs without any prior knowledge of the subject or exercise.
