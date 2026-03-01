MODEL=$1
BASE_URL=${2:-"https://api.openai.com/v1"}
API_KEY=${3:-$OPENAI_UMASS_API_KEY}
MAX_TOKENS=${4:-}
LOG_DIR="logs/${MODEL//\//_}"
export OPENAI_API_KEY=$API_KEY
export BASE_URL=$BASE_URL
mkdir -p $LOG_DIR


# python end_to_end.py --model $MODEL --max-tokens $MAX_TOKENS &> $LOG_DIR/end_to_end.log
# echo "End to end done"


# python dyadic_bottleneck.py --model $MODEL --max-tokens $MAX_TOKENS &> $LOG_DIR/dyadic.log
# echo "Dyadic done"

# python mft_bottleneck.py --model $MODEL  --max-tokens $MAX_TOKENS &> $LOG_DIR/mft.log
# echo "MFT done"

python moral_bottleneck.py --theory-name deontology  --theory-prompt-path prompts/deontology.txt --model $MODEL --num-addn-qns 4  --max-tokens $MAX_TOKENS &> $LOG_DIR/deontology.log
echo "Deontology done"
python moral_bottleneck.py --theory-name utilitarianism  --theory-prompt-path prompts/utilitarianism.txt --model $MODEL --num-addn-qns 4 --max-tokens $MAX_TOKENS &> $LOG_DIR/utilitarianism.log
echo "Utilitarianism done"
python moral_bottleneck.py --theory-name morality_as_cooperation  --theory-prompt-path prompts/morality_as_cooperation.txt --model $MODEL --num-addn-qns 0 --max-tokens $MAX_TOKENS &> $LOG_DIR/morality_as_cooperation.log
echo "Morality as cooperation done"
python moral_bottleneck.py --theory-name virtue_ethics  --theory-prompt-path prompts/virtue_ethics.txt --model $MODEL --num-addn-qns 0 --max-tokens $MAX_TOKENS &> $LOG_DIR/virtue_ethics.log
echo "Virtue ethics done"


python moral_bottleneck.py --theory-name dyadic_reasnoning  --theory-prompt-path prompts/dyadic_reasoning.txt --model $MODEL --num-addn-qns 0 &> $LOG_DIR/dyadic_reasoning.log
echo "Dyadic reasoning done"
python moral_bottleneck.py --theory-name virtue_ethics_reasoning  --theory-prompt-path prompts/virtue_ethics_reasoning.txt --model $MODEL --num-addn-qns 4 &> $LOG_DIR/virtue_ethics_reasoning.log
echo "Virtue ethics reasoning done"



# python moral_bottleneck.py --theory-name dyadic_reasoning  --theory-prompt-path prompts/dyadic_reasoning.txt --model $MODEL --num-addn-qns 0 --api-key $API_KEY --base-url $BASE_URL &> $LOG_DIR/dyadic_reasoning.log
# echo "Dyadic reasoning done"
# python moral_bottleneck.py --theory-name virtue_ethics_reasoning  --theory-prompt-path prompts/virtue_ethics_reasoning.txt --model $MODEL --num-addn-qns 0 --api-key $API_KEY --base-url $BASE_URL &> $LOG_DIR/virtue_ethics_reasoning.log
# echo "Virtue ethics reasoning done"