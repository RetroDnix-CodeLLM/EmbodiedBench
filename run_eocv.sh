#!/bin/bash
#SBATCH -J EmbodiedBench-EOCV
#SBATCH -o logs/eval_log_%j.out
#SBATCH -e logs/eval_log_%j.err
#SBATCH -w gpu16
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:a100-pcie-40gb:1


export DASHSCOPE_API_KEY="sk-ae6624a5b29848ed87132c9c7e8a375c"

cd ../reranker-server

source .venv/bin/activate
vllm serve /home/hyzheng2/QYProjects/models/Qwen/Qwen3.5-2B --tensor-parallel-size 1 --reasoning-parser qwen3 --max_model_len 12800 &
deactivate

VLLM_PID=$!

echo "Waiting for vLLM server to be ready..."

# 轮询接口
timeout=1800
elapsed=0

until [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/v1/models)" = "200" ]; do
    sleep 1
    elapsed=$((elapsed+1))

    if [ "$elapsed" -ge "$timeout" ]; then
        echo "vLLM failed to start within $timeout seconds"
        exit 1
    fi
done

echo "vLLM is ready!"

cd ../EmbodiedBench

source /home/hyzheng2/anaconda3/etc/profile.d/conda.sh
conda activate habitat

exp_name="eocv-$(date +%m%d-%H%M%S)"

EXTRA_EOCV=1 python -m embodiedbench.main env=eb-hab model_name=qwen3-vl-plus exp_name="$exp_name"
