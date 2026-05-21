import sys
import os

# Add the parent directory to path so we can import runner_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__sys__.argv[0] if '__sys__' in globals() else __file__))))
from runner_utils import run_experiment_grid

script_path = "musclesinaction/inference_scripts/retrieval_id_cnn_people.py"

experiment_args = {
    "std": "False",
    "threed": "True",
    "predemg": "True",
}

subjects = [
    "Subject1", "Subject2", "Subject3", "Subject4", "Subject5",
    "Subject6", "Subject7", "Subject8", "Subject9", "Subject10"
]

val_file_map = {
    "Subject5": "valLionel.txt",
    "Subject6": "valMe.txt"
}

train_file_map = {
    "Subject5": "trainLionel.txt"
}

variations = []
for i, sub in enumerate(subjects):
    val_file = val_file_map.get(sub, f"val{sub}.txt")
    train_file = train_file_map.get(sub, f"train{sub}.txt")
    
    variations.append({
        "session_name": f"my_session_clean_{sub}",
        "cuda": i % 8,
        "args": {
            "name": f"generalization_test_cond_{sub}_clean_baseline_perex",
            "resume": f"checkpoints/generalization_test_cond_{sub}_clean/model_100.pth",
            "data_path_train": f"musclesinaction/ablation/generalization_ID_cond_exercises_nn/{train_file}",
            "data_path_val": f"musclesinaction/ablation/generalization_ID_cond_exercises_nn/{val_file}",
        }
    })

if __name__ == "__main__":
    run_experiment_grid(script_path, experiment_args, variations)
