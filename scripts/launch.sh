#!/bin/bash

modes=("language")
learnings=("SFT")
models=("unsloth/gemma-3-4b-it")

for mode in "${modes[@]}"; do
    for model in "${models[@]}"; do
        sbatch train_sft.sh "$mode" "$model"
    done
done
