import os
import json
from pathlib import Path

data_base = [
    "/data/home/hyzheng/Projects/PcT/EmbodiedBench/running/eb_habitat/qwen3.6-plus_react-0420-1435",
    "/data/home/hyzheng/Projects/PcT/EmbodiedBench/running/eb_habitat/qwen3.6-plus_eocv-0415-0009"
    # "/home/hyzheng2/QYProjects/EmbodiedBench/running/eb_habitat/qwen3-vl-plus_eocv-0401-130226",
    # "/home/hyzheng2/QYProjects/EmbodiedBench/running/eb_habitat/qwen3-vl-plus_multi-step-0401-022525",
    # "/home/hyzheng2/QYProjects/EmbodiedBench/running/eb_habitat/qwen3-vl-plus_react-0401-022525"
]

for base in data_base:
    print(f"Processing base directory: {base}")
    task_split = os.listdir(base)
    for task in task_split:
        task_split.sort()  # Ensure consistent order
        process_cnt = 0
        item_cnt = 0
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
                process_cnt += task_progress
                item_cnt += 1
                print(f"Split '{task}' Episode '{log.stem}': Progress: {task_progress : .2f}")
        print(f"Split '{task}': Processed {item_cnt} items, Average Progress: {process_cnt / item_cnt : .2f}")