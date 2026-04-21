import os
import json
from pathlib import Path

data_base = [
    "running/eb_habitat/qwen3.6-plus-react",
    "running/eb_habitat/qwen3.6-plus-eocv",
    "running/eb_habitat/qwen3.6-plus_multi-step-0415-0940"
]

for base in data_base:
    print(f"\nProcessing base directory: {base}")
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

        print(f"Task: {task:<20}, Progress: {task_progress:<10.4f}, Input Tokens: {input_tokens:<10}, Output Tokens: {output_tokens:<10}, Avg Tokens: {int(input_tokens / 6 + output_tokens):<10d}")