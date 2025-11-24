models=("stage1_models/SFT_gemma_27b_language16")

for model in "${models[@]}"; do
    sbatch benchmark.sh "$model"
done

# models=("../stage1_models/SFT_gemma_4b_language16" "../stage1_models/SFT_gemma_12b_language16" "../stage1_models/SFT_Qwen2.5_7B_language16" "../stage1_models/SFT_Qwen2.5_32B_language16")

# for model in "${models[@]}"; do
#     sbatch live_code_bench.sh "$model"
# done







# for model in "${models[@]}"; do
#     sbatch benchmark.sh "$model"
# done

# for model in "${models[@]}"; do
#     sbatch benchmark.sh "$model"
# done
