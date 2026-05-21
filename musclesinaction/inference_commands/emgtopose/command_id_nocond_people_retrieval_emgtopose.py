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
    "data_path_train": "musclesinaction/ablation/generalization_ID_nocond_people/train.txt",
}

subjects = [
    "Subject1", "Subject2", "Subject3", "Subject4", "Subject5",
    "Subject6", "Subject7", "Subject8", "Subject9", "Subject10"
]

variations = []
for i, sub in enumerate(subjects):
    variations.append({
        "session_name": f"my_session_clean_{sub}",
        "cuda": i % 8,
        "args": {
            "name": f"generalization_test_cond_{sub}_clean_baseline_perex",
            "data_path_val": f"musclesinaction/ablation/generalization_ID_nocond_people/val{sub}.txt",
        }
    })

if __name__ == "__main__":
    run_experiment_grid(script_path, experiment_args, variations)
