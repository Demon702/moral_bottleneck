MODEL=$1
LOG_DIR="demographic/circumstance_logs/${MODEL//\//_}"
mkdir -p $LOG_DIR

export API_KEY='921116a30c5af77c5d6457ff9d96aed8ed1bde649f3072f3af0dc59076e1a621'
export BASE_URL='https://api.together.xyz/v1'
python demographic/end_to_end.py --model $MODEL --all-circumstances > $LOG_DIR/end_to_end.log 2>&1
python demographic/dyadic_bottleneck.py --model $MODEL  --all-circumstances > $LOG_DIR/dyadic.log 2>&1
# # python demographic/dyadic_bottleneck_two_step.py --model $MODEL  > $LOG_DIR/dyadic_two_step.log 2>&1
python demographic/mft_bottleneck.py --model $MODEL --all-circumstances  > $LOG_DIR/mft.log 2>&1
# # python demographic/mft_bottleneck_two_step.py --model $MODEL  > $LOG_DIR/mft_two_step.log 2>&1
# python demographic/cot_bottleneck.py --model $MODEL  --all-circumstances > $LOG_DIR/cot.log 2>&1


