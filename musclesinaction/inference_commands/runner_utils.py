import os
import time
import argparse

def run_in_tmux(session_name, command, conda_env=None):
    """
    Runs a command in a new background tmux session, optionally activating a conda environment.
    """
    if conda_env:
        activate_cmd = f"conda activate {conda_env}"
        full_cmd = f'tmux new-session -d -s {session_name} \\; send-keys "{activate_cmd}" Enter \\; send-keys "{command}" Enter'
    else:
        full_cmd = f'tmux new-session -d -s {session_name} \\; send-keys "{command}" Enter'
    
    print(f"Starting tmux session: {session_name}")
    print(f"Command: {command}")
    os.system(full_cmd)

def run_experiment_grid(script_path, experiment_args, variations, default_env="musclesinaction"):
    """
    Main entry point for running a grid of experiments.
    """
    parser = argparse.ArgumentParser(description="Run MIA inference experiments.")
    parser.add_argument("--env", type=str, default=default_env, help=f"Conda environment to use (default: {default_env})")
    parser.add_argument("--sleep", type=int, default=20, help="Sleep time between session starts (default: 20)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--gpu-offset", type=int, default=0, help="Offset for GPU indexing")
    
    # Allow passing any additional args to override defaults
    args, unknown = parser.parse_known_args()

    for i, var in enumerate(variations):
        session_name = var.get('session_name', f"mia_job_{i}")
        cuda_device = var.get('cuda', (i + args.gpu_offset) % 8)
        
        # Build command
        # We add PYTHONPATH=. to ensure the project root is in the path for imports
        cmd_parts = [f"CUDA_VISIBLE_DEVICES={cuda_device}", "PYTHONPATH=.", "python", script_path]
        
        # Merge common and specific args
        all_params = experiment_args.copy()
        all_params.update(var.get('args', {}))
        
        for k, v in all_params.items():
            if isinstance(v, bool):
                cmd_parts.append(f"--{k} {v}")
            else:
                cmd_parts.append(f"--{k} {v}")
            
        full_python_cmd = " ".join(cmd_parts)
        
        if args.dry_run:
            print(f"\n[DRY RUN] Session: {session_name}")
            print(f"Environment: {args.env}")
            print(f"Command: {full_python_cmd}")
        else:
            run_in_tmux(session_name, full_python_cmd, conda_env=args.env)
            if i < len(variations) - 1:
                time.sleep(args.sleep)
