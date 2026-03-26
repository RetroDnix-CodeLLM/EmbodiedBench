import json
from pathlib import Path

# data_base_path = "/home/hyzheng2/QYProjects/EmbodiedBench/running/eb_alfred/gpt-4o_multi-step/base/results"
data_base_path = "/home/hyzheng2/QYProjects/EmbodiedBench/running/eb_alfred/gpt-4o_react/base/results"
log_files = list(Path(data_base_path).glob("*_final_res.json"))
log_files.sort()

for log_file in log_files:
    with open(log_file, "r") as f:
        data = json.load(f)
    
    task_progress = data["task_progress"]

    print(f"Task: {log_file.stem}, Progress: {task_progress}")