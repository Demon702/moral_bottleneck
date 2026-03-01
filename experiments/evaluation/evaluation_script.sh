# THEORIES=("utilitarianism" "deontology" "morality_as_cooperation" "virtue_ethics")
THEORIES=("dyadic" "mft" "end_to_end_all" "deontology" "utilitarianism" "morality_as_cooperation" "virtue_ethics")
echo "Evaluating moral theories"

# MODELS=("gpt-4o" "gpt-3.5-turbo" "meta_llama3-70b-instruct" "mistralai_Mixtral-8x22B-Instruct-v0.1" "o3-mini" "deepseek-ai_DeepSeek-R1")
MODELS=("deepseek-ai_DeepSeek-R1")
for MODEL in "${MODELS[@]}"; do
    for THEORY in "${THEORIES[@]}"; do
        echo "----------------------------------------------------------------------------------------------------------"
        echo "Evaluating $THEORY for $MODEL\n\n\n"
        mkdir -p evaluation_logs/$MODEL
        # echo "Evaluating model gpt-3.5-turbo"
        # python evaluation_moral_theory.py --model gpt-3.5-turbo --theory $THEORY > evaluation_logs/gpt-3.5-turbo/$THEORY.txt 2>&1

        python evaluation_moral_theory.py --model $MODEL --theory $THEORY --add_reg_scores > evaluation_logs/$MODEL/${THEORY}.txt 2>&1

        # echo "Evaluating model meta/llama3-70b-instruct"
        # python evaluation_moral_theory.py --model meta/llama3-70b-instruct --theory $THEORY > evaluation_logs/meta_llama3-70b-instruct/$THEORY.txt 2>&1

        # echo "Evaluating model mistralai/mixtral-8x22b-instruct-v0.1"
        # python evaluation_moral_theory.py --model mistralai/mixtral-8x22b-instruct-v0.1 --theory $THEORY > evaluation_logs/mistralai_mixtral-8x22b-instruct-v0.1/$THEORY.txt 2>&1
    done
done