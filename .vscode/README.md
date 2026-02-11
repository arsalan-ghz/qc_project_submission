# VS Code helpers for this project

Quick notes to run and debug `run_qcardest_style_training.py`.

1) Select the Python interpreter for the workspace (choose the `qc_project` Conda environment).

2) Debug using the provided launch configurations: open the Run view (Ctrl+Shift+D), select a configuration ("Debug run_qcardest_style_training (example)"), then press F5.

3) Run the prepared task (no debug): open the Command Palette (Ctrl+Shift+P) → `Tasks: Run Task` → choose `Run training (conda)`.

4) To change CLI arguments for debugging, edit `.vscode/launch.json` (the `args` array). To change the task command, edit `.vscode/tasks.json`.

Notes:
- The task uses `conda run -n qc_project` so you must have Conda available in your PATH in the VS Code terminal.
- The debug configuration uses `integratedTerminal` so output and interactive prompts appear in the VS Code terminal.
