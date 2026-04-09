import os
import json
from pathlib import Path

data_base = [
    "running/eb_habitat/qwen3-vl-plus_eocv-0408-133442_ord",
    "running/eb_habitat/qwen3-vl-plus_react-0408-221710_ord",
    "running/eb_habitat/qwen3-vl-plus_multi-step-0401-140724_ord"
]

for base in data_base:
    print(f"Processing base directory: {base}")
    task_split = os.listdir(base)
    task_split.sort()  # Ensure consistent order
    for task in task_split:
        log_dir = Path(base) / task / "results/summary.json"
        if not os.path.exists(log_dir):
            print(f"Log file not found for task: {task}")
            continue
        log = json.load(open(log_dir, "r"))
        task_progress = log["task_progress"]

        usage_dir = Path(base) / task / "results/total_token_usage.json"
        usage = json.load(open(usage_dir, "r"))
        input_tokens = usage["input_tokens"]
        output_tokens = usage["output_tokens"]

        print(f"Task: {task}, Progress: {task_progress}, Input Tokens: {input_tokens}, Output Tokens: {output_tokens}, Avg Tokens: {input_tokens * 0.1 + output_tokens}")