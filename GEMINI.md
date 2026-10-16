# Antigravity Workspace Rules & Safety Boundaries

## 1. Strict Filesystem Boundary
- **Never exit this directory**: All operations, file edits, script creations, log writings, and outputs MUST remain strictly inside `/home/ee23btech11057/PlayingWithDeepImagePrior/`.
- Do not inspect, read, write, or create files in `/home/ee23btech11057/` (outside this repo), `/tmp`, `/var`, or any external directories.
- All experiment artifacts must be saved inside subdirectories of this project (e.g. `output_swin_test/`, `output_dip_sr/`).

## 2. Autonomous Actions (No Permission Needed)
The agent is authorized to proceed autonomously for:
- Writing and modifying research/experiment code within this directory.
- Running Python training and evaluation scripts within `.venv`.
- Launching background runs/screen sessions for experiments inside this directory.
- Analyzing outputs, logs, WandB metrics, and plotting figures in `output_*/`.

## 3. Mandatory User Confirmation (ALWAYS Ask Permission)
The agent MUST pause and explicitly ask the user before performing any of the following:
- **Destructive filesystem actions**: Deleting existing files, clearing directories (`rm`, `git clean`).
- **Git modifications**: Branch creation/switching, `git reset`, `git revert`, committing, or pushing to remote.
- **Environment mutations**: Installing, upgrading, or removing packages in `.venv` (`pip install`, `uv add`, etc.).
- **Process termination**: Killing running screen sessions, background processes, or GPU tasks that were not launched in the immediate turn.
- **Any action touching external paths**: If any command ever requires accessing a path outside `/home/ee23btech11057/PlayingWithDeepImagePrior/`, explicitly ask first.
