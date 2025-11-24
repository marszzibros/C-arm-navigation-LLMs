#!/bin/bash

#SBATCH --partition=hgnodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=01-23:59:59
#SBATCH --job-name=vlm_ft
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

export UNSLOTH_DISABLE_PATCHING=1

source ~/.bashrc
module load cuda/12.6.2
conda activate xray_llm

cd ${SLURM_SUBMIT_DIR}

model=$1
mode=$2

python3 /gpfs3/scratch/jjung2/vlm_finetuning/tester.py $model $mode
