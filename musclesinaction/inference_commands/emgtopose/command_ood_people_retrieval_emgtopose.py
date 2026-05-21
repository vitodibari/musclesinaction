import sys
import os

# Add the parent directory to path so we can import runner_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__sys__.argv[0] if '__sys__' in globals() else __file__))))
from runner_utils import run_experiment_grid

script_path = "musclesinaction/inference_scripts/retrieval_id_nocond_exercises_emgtopose.py"

experiment_args = {
    "std": "False",
    "threed": "True",
    "predemg": "True",
}

subjects = [
    "Subject2", "Subject10", "Subject9", "Subject7", "David",
    "Subject3", "Subject4", "Subject5", "Subject6", "Subject8",
    "general"
]

variations = []
for i, sub in enumerate(subjects):
    if sub == "general":
        train_path = "musclesinaction/ablation/generalizationFinal/train.txt"
        val_path = "musclesinaction/ablation/generalizationFinal/val.txt"
    else:
        train_path = f"musclesinaction/ablation/generalizationpeople/train_{sub}.txt"
        val_path = f"musclesinaction/ablation/generalizationpeople/val_{sub}.txt"
        
    variations.append({
        "session_name": f"my_session_clean_{sub}",
        "cuda": i % 8,
        "args": {
            "name": f"generalization_test_cond_{sub}_clean_baseline",
            "data_path_train": train_path,
            "data_path_val": val_path,
        }
    })

if __name__ == "__main__":
    run_experiment_grid(script_path, experiment_args, variations)
