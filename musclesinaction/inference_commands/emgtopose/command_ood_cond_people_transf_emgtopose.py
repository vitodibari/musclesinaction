import sys
import os

# Add the parent directory to path so we can import runner_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__sys__.argv[0] if '__sys__' in globals() else __file__))))
from runner_utils import run_experiment_grid

script_path = "musclesinaction/inference_scripts/inference_id_transf_cond_exercises_emgtopose.py"

experiment_args = {
    "std": "False",
    "threed": "True",
    "cond": "True",
    "predemg": "False",
}

subjects = [
    "Subject2", "Subject10", "Subject9", "Subject7", "Subject1",
    "Subject3", "Subject4", "Subject5", "Subject6", "Subject8"
]

variations = []
for i, sub in enumerate(subjects):
    variations.append({
        "session_name": f"my_session_clean_{sub}",
        "cuda": i % 8,
        "args": {
            "name": f"generalization_new_nocond_{sub}_clean_baseline",
            "resume": f"checkpoints/generalization_new_cond_{sub}_clean_emgtopose/model_50.pth",
            "data_path_train": f"musclesinaction/ablation/generalizationpeople/train_{sub}.txt",
            "data_path_val": f"musclesinaction/ablation/generalizationpeople/val_{sub}.txt",
        }
    })

if __name__ == "__main__":
    run_experiment_grid(script_path, experiment_args, variations)
