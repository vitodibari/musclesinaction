import os
import pdb
import time

import os

# We remove all the tmux-specific commands and just keep the core execution
command = (
    "CUDA_VISIBLE_DEVICES=1 python /home/laboratorio/vitodibari/musclesinaction/musclesinaction/inference_scripts/inference_id_transf_cond_exercises_posetoemg.py "
    "--name generalization_new_cond_SlowSkater_clean_baseline_perex "
    "--std False "
    "--threed True "
    "--predemg True "
    "--cond True "
    "--resume pretrained-checkpoints/generalization_new_cond_clean_posetoemg/model_100.pth "
    "--data_path_train musclesinaction/ablation/generalization_ID_nocond_exercises/train.txt "
    "--data_path_val musclesinaction/ablation/generalization_ID_nocond_exercises/valSlowSkater.txt"
)

# This executes the command directly in your current shell
os.system(command)