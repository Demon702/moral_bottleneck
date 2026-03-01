# Move the six theory tsv files for each model to a new folder called "theory"

MODELS=("gpt-4o" "gpt-3.5-turbo" "o3-mini" "meta_llama3-70b-instruct" "mistralai_mixtral-8x22b-instruct-v0.1" "deepseek-ai_DeepSeek-R1")
THEORIES=("deontology" "utilitarianism" "morality_as_cooperation" "virtue_ethics" "dyadic" "mft" "end_to_end_all")

mkdir -p ../theory_results
for model in "${MODELS[@]}"; do
    mkdir -p "../theory_results/${model}"
    for theory in "${THEORIES[@]}"; do
        cp "../results/${model}/${theory}.tsv" "../theory_results/${model}/${theory}.tsv"
        # display error if the file does not exist, but continue to the next file
        if [ ! -f "../results/${model}/${theory}_with_reg_scores.tsv" ]; then
            echo "Error: ${model}/${theory}_with_reg_scores.tsv does not exist"
        else
            cp "../results/${model}/${theory}_with_reg_scores.tsv" "../theory_results/${model}/${theory}_with_suvervised_scores.tsv"
        fi
    done
done