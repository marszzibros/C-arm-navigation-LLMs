#!/bin/bash
modes=("test" "classification")
for mode in "${modes[@]}"; do

    # models=("stage1_models/SFT_gemma_27b_language16" "stage1_models/SFT_gemma_12b_language16" "stage1_models/SFT_gemma_4b_language16")

    # for model in "${models[@]}"; do
    #     sbatch tester1.sh "$model" "$mode"
    # done

    # models=("stage1_models/SFT_all_linear_Qwen2.5_7B_language16" "stage1_models/SFT_all_linear_Qwen2.5_32B_language16")


    # for model in "${models[@]}"; do
    #     sbatch tester1.sh "$model" "$mode"
    # done

    # models=("stage1_models/SFT_Qwen2.5_7B_language16" "stage1_models/SFT_Qwen2.5_32B_language16")

    # for model in "${models[@]}"; do
    #     sbatch tester1.sh "$model" "$mode"
    # done

    # models=("unsloth/gemma-3-4b-it" "unsloth/gemma-3-12b-it" "unsloth/gemma-3-27b-it")

    # for model in "${models[@]}"; do
    #     sbatch tester1.sh "$model" "$mode"
    # done

    # models=("unsloth/Qwen2.5-VL-7B-Instruct" "unsloth/Qwen2.5-VL-32B-Instruct")

    # for model in "${models[@]}"; do
    #     sbatch tester1.sh "$model" "$mode"
    # done

    # models=("google/medgemma-4b-it" "google/medgemma-27b-it")

    # for model in "${models[@]}"; do
    #     sbatch tester1.sh "$model" "$mode"
    # done




#     models=("stage1_models/SFT_gemma_4b_language16" "stage1_models/SFT_gemma_12b_language16" "stage1_models/SFT_Qwen2.5_7B_language16")

#     for model in "${models[@]}"; do
#         sbatch tester1.sh "$model" "$mode"
#     done

#     models=("stage1_models/SFT_all_linear_gemma_4b_language16" "stage1_models/SFT_all_linear_gemma_12b_language16" "stage1_models/SFT_all_linear_Qwen2.5_7B_language16")

#     for model in "${models[@]}"; do
#         sbatch tester1.sh "$model" "$mode"
#     done

    # models=("unsloth/Qwen2.5-VL-32B-Instruct" "stage1_models/SFT_Qwen2.5_32B_language16" "stage1_models/SFT_all_linear_Qwen2.5_32B_language16")

    # for model in "${models[@]}"; do
    #     sbatch tester1.sh "$model" "$mode"
    # done

    # models=("stage1_models/SFT_all_linear_Qwen2.5_32B_language16" "google/medgemma-4b-it" "google/medgemma-27b-it")

    # for model in "${models[@]}"; do
    #     sbatch tester1.sh "$model" "$mode"
    # done

    # models=("/gpfs1/llm/llama-4.0-hf/Llama-4-Scout-17B-16E-Instruct")

    # for model in "${models[@]}"; do
    #     sbatch tester1.sh "$model"
    # done
done
