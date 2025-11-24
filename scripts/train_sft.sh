#!/bin/bash

#SBATCH --partition=hgnodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --time=01-23:59:59
#SBATCH --job-name=vlm_ft
#SBATCH --mail-user=jjung2@uvm.edu
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

source ~/.bashrc
module load cuda/12.6.2
conda activate xray_llm

cd ${SLURM_SUBMIT_DIR}

mode=$1
model=$2

python3 /gpfs3/scratch/jjung2/vlm_finetuning/train.py mode=$mode train=SFT model_id=$model
