#!/bin/bash

#SBATCH --partition=nvgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --time=01-23:59:59
#SBATCH --job-name=vlm_bench
#SBATCH --mail-type=ALL


cd ${SLURM_SUBMIT_DIR}

source ~/.bashrc
module load cuda/12.6.2
conda activate xray_llm

cd ${SLURM_SUBMIT_DIR}


save_dir="eval_results/"
global_record_file="eval_results/eval_record_collection.csv"
model=$1
selected_subjects="all"
gpu_util=0.8
export CUDA_VISIBLE_DEVICES=0

python3 evaluate_from_local.py \
                 --selected_subjects $selected_subjects \
                 --save_dir $save_dir \
                 --model $model \
                 --global_record_file $global_record_file \
                 --gpu_util $gpu_util