import json

RESULTS = [
    "/data/home/hyzheng/Projects/PcT/EmbodiedBench/running/eb_habitat/qwen3.6-plus-eocv/base/results/summary.json",
    "/data/home/hyzheng/Projects/PcT/EmbodiedBench/running/eb_habitat/qwen3.6-plus-eocv/common_sense/results/summary.json",
    "/data/home/hyzheng/Projects/PcT/EmbodiedBench/running/eb_habitat/qwen3.6-plus-eocv/complex_instruction/results/summary.json",
    "/data/home/hyzheng/Projects/PcT/EmbodiedBench/running/eb_habitat/qwen3.6-plus-eocv/long_horizon/results/summary.json",
    "/data/home/hyzheng/Projects/PcT/EmbodiedBench/running/eb_habitat/qwen3.6-plus-eocv/spatial_relationship/results/summary.json",
    "/data/home/hyzheng/Projects/PcT/EmbodiedBench/running/eb_habitat/qwen3.6-plus-eocv/visual_appearance/results/summary.json"
]

for result in RESULTS:
    with open(result, "r") as f:
        data = json.load(f)
    input_tokens = data["input_tokens"]
    output_tokens = data["output_tokens"]
    total_tokens = input_tokens / 6 + output_tokens

    eocv_input_tokens = data["eocv_input_tokens"]
    eocv_output_tokens = data["eocv_output_tokens"]
    eocv_total_tokens = eocv_input_tokens / 6 + eocv_output_tokens

    percent = eocv_total_tokens / (total_tokens + eocv_total_tokens)
    print(f"Total tokens: {total_tokens:.2f}, EoCV tokens: {eocv_total_tokens:.2f}, Percent: {percent:.2%}")