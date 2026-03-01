MODEL=$1
LOG_DIR="demographic/circumstance_logs/${MODEL//\//_}"
mkdir -p $LOG_DIR
export API_KEY=$OPENAI_API_KEY
unset BASE_URL
# python end_to_end.py --model $MODEL --use-together-api  > $LOG_DIR/end_to_end.log
# python dyadic_bottleneck.py --model $MODEL --use-together-api  > $LOG_DIR/dyadic.log
# python dyadic_bottleneck_two_step.py --model $MODEL --use-together-api  > $LOG_DIR/dyadic_two_step.log
# python mft_bottleneck.py --model $MODEL --use-together-api  > $LOG_DIR/mft.log
# python mft_bottleneck_two_step.py --model $MODEL --use-together-api  > $LOG_DIR/mft_two_step.log
# python cot_bottleneck.py --model $MODEL --use-together-api  > $LOG_DIR/cot.log



python demographic/end_to_end.py --model $MODEL --all-circumstances > $LOG_DIR/end_to_end.log 2>&1
python demographic/dyadic_bottleneck.py --model $MODEL --all-circumstances > $LOG_DIR/dyadic.log 2>&1
# python demographic/dyadic_bottleneck_two_step.py --model $MODE > $LOG_DIR/dyadic_two_step.log 2>&1
python demographic/mft_bottleneck.py --model $MODEL --all-circumstances > $LOG_DIR/mft.log 2>&1
# python demographic/mft_bottleneck_two_step.py --model $MODEL  > $LOG_DIR/mft_two_step.log 2>&1
# python demographic/cot_bottleneck.py --model $MODEL --all-circumstances > $LOG_DIR/cot.log 2>&1


