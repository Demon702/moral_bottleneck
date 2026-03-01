# MODELS=("gpt-4o" "gpt-3.5-turbo" "meta_llama3-70b-instruct" "mistralai_Mixtral-8x22B-Instruct-v0.1" "o3-mini")
MODELS=("deepseek-ai_DeepSeek-R1")
for MODEL in "${MODELS[@]}"; do
    echo "----------------------------------------------------------------------------------------------------------"
    echo "Evaluating end-to-end regression for $MODEL\n\n\n"
    mkdir -p evaluation_logs/$MODEL
    python evaluate_end_to_end_regression.py --model $MODEL > evaluation_logs/$MODEL/end_to_end_regression.txt 2>&1
done
