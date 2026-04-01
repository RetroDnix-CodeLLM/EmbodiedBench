import os
import json
from pathlib import Path

data_base = [
    "/home/hyzheng2/QYProjects/EmbodiedBench/running/eb_habitat/qwen3-vl-plus_eocv-0401-093507",
    "/home/hyzheng2/QYProjects/EmbodiedBench/running/eb_habitat/qwen3-vl-plus_multi-step-0401-022525",
    "/home/hyzheng2/QYProjects/EmbodiedBench/running/eb_habitat/qwen3-vl-plus_react-0401-022525"
]

for base in data_base:
    print(f"Processing base directory: {base}")
    task_split = os.listdir(base)
    task_split.sort()  # Ensure consistent order
    for task in task_split:
        log_dir = Path(base) / task / "results"
        if not os.path.exists(log_dir):
            print(f"Log file not found for task: {task}")
            continue
        
        logs = list(log_dir.glob("*_res.json"))
        logs.sort()
        for log in logs:
            log_data = json.load(open(log, "r"))
            if "task_progress" in log_data:
                task_progress = log_data["task_progress"]
                print(f"Split '{task}' Episode '{log.stem}': Progress: {task_progress : .2f}")