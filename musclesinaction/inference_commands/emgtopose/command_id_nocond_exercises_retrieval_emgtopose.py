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
    "data_path_train": "musclesinaction/ablation/generalization_ID_nocond_exercises/train.txt",
}

exercises = [
    "Running", "RonddeJambe", "LegCross", "LegBack", "KneeKick", 
    "JumpingJack", "HookPunch", "HighKick", "FrontPunch", "FrontKick", 
    "ElbowPunch", "Shuffle", "SideLunges", "SlowSkater", "Squat"
]

variations = []
for i, ex in enumerate(exercises):
    session_prefix = "val" if ex in ["SideLunges", "SlowSkater", "Squat"] else ""
    session_name = f"my_session_clean_{session_prefix}{ex}"
    
    variations.append({
        "session_name": session_name,
        "cuda": i % 8,
        "args": {
            "name": f"generalization_test_cond_{ex}_clean_baseline_perex",
            "data_path_val": f"musclesinaction/ablation/generalization_ID_nocond_exercises/val{ex}.txt",
        }
    })

if __name__ == "__main__":
    run_experiment_grid(script_path, experiment_args, variations)
