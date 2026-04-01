#!/bin/bash
#SBATCH -J EmbodiedBench-EOCV
#SBATCH -o logs/eval_log_%j.out
#SBATCH -e logs/eval_log_%j.err
#SBATCH -w gpu16
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:a100-pcie-40gb:1

source /home/hyzheng2/anaconda3/etc/profile.d/conda.sh
conda activate habitat

export DASHSCOPE_API_KEY="sk-ae6624a5b29848ed87132c9c7e8a375c"

exp_name="react-$(date +%m%d-%H%M%S)"

EXTRA_ONE_STEP=1 python -m embodiedbench.main env=eb-hab model_name=qwen3-vl-plus exp_name="$exp_name"
