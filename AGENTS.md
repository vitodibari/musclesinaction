# Agent Guide: Muscles in Action (ICCV 2023)

Welcome, AI Agent or Developer! This guide provides a comprehensive overview of the **Muscles in Action (MIA)** codebase, project structure, dataset details, and the core scientific concepts behind the paper. It is designed to get you up to speed quickly for training, evaluation, and code modifications.

---

## 1. Research Background & Paper Overview

The project implements the paper **"Muscles in Action" (ICCV 2023)** by Mia Chiquier and Carl Vondrick.
* **Core Goal**: Learn a **bidirectional mapping** between visible human motion (3D/2D skeletons) and internal muscle activity (surface electromyography, or sEMG).
* **Key Components**:
  * **Motion-to-Muscle (Pose-to-EMG) Encoder**: Predicts the sEMG muscle activation sequence over time from an input video or skeleton keypoints.
  * **Muscle-to-Motion (EMG-to-Pose) Decoder**: Reconstructs the 3D human pose trajectory over time from sEMG activation signals.
  * **Muscle Modification / Editing**: Edits predicted muscle activations (scaling them up or down) and passes them to the decoder to recommend joint movements that target specific muscle groups.

### The MIA Dataset
* **Scale**: 12.5 hours of synchronized video and sEMG signals.
* **Subjects**: 10 subjects (5 male, 5 female).
* **Exercises**: 15 distinct physical exercises (e.g., Squat, Slow Skater, Jumping Jack, Elbow Punch, Hook Punch, Running).
* **Sensors**: 8 sEMG sensors measuring the bio-electric energy of the following muscles:
  1. Left Biceps Brachii (`leftbicep`)
  2. Right Biceps Brachii (`rightbicep`)
  3. Left Latissimus Dorsi (`leftham` - labeled as `leftglutt`/`laterals` conceptually)
  4. Right Latissimus Dorsi (`rightham` - labeled as `rightglutt`/`laterals` conceptually)
  5. Left Quadriceps (`leftquad`)
  6. Right Quadriceps (`rightquad`)
  7. Left Hamstring (conceptual, mapped to indices)
  8. Right Hamstring (conceptual, mapped to indices)

---

## 2. Project Architecture

The directory structure is organized as follows:

```
musclesinaction/
├── config/                  # Configuration YAML files (train.yaml, inference.yaml)
│   └── args.py              # CLI Argument parsing and verification
├── dataloader/
│   └── data.py              # PyTorch Dataset (MyMuscleDataset) and dataloaders
├── models/
│   ├── modelemgtopose.py    # Transformer architecture to predict Pose from EMG
│   └── modelposetoemg.py    # Transformer architecture to predict EMG from Pose
├── pipeline.py              # Forward pass execution wrapper for DataParallel & loss computation
├── train.py                 # Main training and validation supervisor script
├── inference_scripts/       # Scripts running evaluation logic for specific settings
├── inference_commands/      # Grid execution scripts utilizing tmux to launch parallel runs
│   └── runner_utils.py      # Utilities for grid runs (run_experiment_grid, run_in_tmux)
├── ablation/                # Predefined train/val text splits for generalization experiments
└── README.md                # General setup and execution guide
```

### Key Modules Walkthrough

* **`train.py`**: Handles initialization (seeding, loaders, optimizer, MultiStepLR scheduler), model checkpointing (saving to `checkpoints/`), and epoch loops.
* **`pipeline.py`**: Houses `MyTrainPipeline` wrapping the model. Supports both 3D skeletons (x, y, z joint coordinates) and 2D projections. Computing mean squared error (`MSELoss`) between predicted and ground-truth EMG values or Pose keypoints.
* **`dataloader/data.py`**:
  * Loads `.npy` arrays for EMG, skeletons (2D and 3D), and bounding boxes.
  * Translates raw values based on subject-specific calibration constants (`themax`, `themin`).
  * Employs subject identity conditioning: concatenating a personal identifier coordinate (`condval`) to coordinate spaces.
* **`models/`**:
  * Contains local feature extractors (1D temporal convolutions to embed the sequence) followed by a global sequence-to-sequence Transformer encoder with 4 layers and 8 attention heads.

---

## 3. Core Concepts & Terminology

Understanding the vocabulary is critical when analyzing configs, filenames, or logs:

1. **Model Approach**:
   * `transf`: Uses a neural network built around a **Transformer** encoder to perform predictions.
   * `retrieval`: A baseline that uses a **Nearest Neighbor (L2)** search to return matching training sequences.
2. **Generalization Type**:
   * `id` (In-Distribution): The test subject and exercise are present in the training set (evaluated on unseen time segments).
   * `ood` (Out-of-Distribution): Zero-shot testing. Evaluates on subjects (`people`) or exercises (`exercises`) completely excluded from the training split.
3. **Subject/Action Focus**:
   * `people`: Subject-to-Subject generalization experiments.
   * `exercises`: Exercise-to-Exercise generalization experiments.
4. **Conditioning**:
   * `cond` (Conditional): Models are explicitly given the identity coordinate of the subject or exercise.
   * `nocond` (Non-conditional): The model receives EMG or Pose signals blindly without subject/exercise context.

---

## 4. Run Configurations

### Training Models
To launch model training, configure options in `musclesinaction/configs/train.yaml` or pass parameters to `train.py`.
```bash
python musclesinaction/train.py --predemg True  # Train Pose-to-EMG model
python musclesinaction/train.py --predemg False # Train EMG-to-Pose model
```

### Running Inference Grids
Inference grid runners inside `musclesinaction/inference_commands` automatically spawn background training/evaluation routines on separate CUDA devices in parallel using `tmux` sessions.

For example, to run an In-Distribution, Conditional, Exercise-focused retrieval evaluation for EMG-to-Pose:
```bash
python musclesinaction/inference_commands/emgtopose/command_id_cond_exercises_retrieval_emgtopose.py
```

---

## 5. Implementation Tips & Gotchas for AI Agents

* **Tmux Spawning**: Calling inference commands spawns tmux sessions in the background. If you want to debug or inspect output directly without tmux, use the `--dry-run` flag to output the raw commands:
  ```bash
  python musclesinaction/inference_commands/emgtopose/command_..._emgtopose.py --dry-run
  ```
  Then run the returned python command directly in your shell.
* **Device Offset**: In runner scripts, CUDA index allocation is cyclic (`i % 8` or `(i + args.gpu_offset) % 8`). If your host machine has a different number of GPUs, pass `--gpu-offset` or adjust the target devices appropriately.
* **Paths**: Ensure that paths mapped in `train.yaml`/`inference.yaml` align with where you placed the downloaded `MIADatasetOfficial` and SMPL/vibe checkpoint files.
* **Batch Size**: The dataset uses `args.bs = 1` for evaluations to evaluate sequence-by-sequence metrics. Ensure you respect this parameter when debugging test outputs.
