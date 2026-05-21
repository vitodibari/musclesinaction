import sys
import os

# Add the parent directory to path so we can import runner_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__sys__.argv[0] if '__sys__' in globals() else __file__))))
from runner_utils import run_experiment_grid

script_path = "musclesinaction/inference_scripts/retrieval_id_transf_cond_exercises.py"

experiment_args = {
    "std": "False",
    "threed": "True",
    "cond": "True",
    "predemg": "True",
    "resume": "pretrained-checkpoints/generalization_new_cond_clean_posetoemg/model_100.pth",
    "data_path_train": "musclesinaction/ablation/generalization_ID_nocond_people/train.txt",
}

subjects = [
    "Subject1", "Subject2", "Subject3", "Subject4", "Subject5",
    "Subject6", "Subject7", "Subject8", "Subject9", "Subject10"
]

# Mapping subjects to their specific validation files if they differ from the name
val_file_map = {
    "Subject5": "valLionel.txt"
}

variations = []
for i, sub in enumerate(subjects):
    val_file = val_file_map.get(sub, f"val{sub}.txt")
    
    variations.append({
        "session_name": f"my_session_clean_{sub}",
        "cuda": i % 8,
        "args": {
            "name": f"generalization_new_cond_{sub}_clean_baseline_perex",
            "data_path_val": f"musclesinaction/ablation/generalization_ID_nocond_people/{val_file}",
        }
    })

if __name__ == "__main__":
    run_experiment_grid(script_path, experiment_args, variations)
