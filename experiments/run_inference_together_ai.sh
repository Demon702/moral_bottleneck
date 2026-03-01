MODEL=$1
LOG_DIR="logs/${MODEL//\//_}"
mkdir -p $LOG_DIR

export API_KEY='921116a30c5af77c5d6457ff9d96aed8ed1bde649f3072f3af0dc59076e1a621'
export BASE_URL='https://api.together.xyz/v1'
python end_to_end.py --model $MODEL --use-together-api  > $LOG_DIR/end_to_end.log 2>&1
python dyadic_bottleneck.py --model $MODEL --use-together-api  > $LOG_DIR/dyadic.log 2>&1
# python dyadic_bottleneck_two_step.py --model $MODEL --use-together-api  > $LOG_DIR/dyadic_two_step.log 2>&1
python mft_bottleneck.py --model $MODEL --use-together-api  > $LOG_DIR/mft.log 2>&1
# python mft_bottleneck_two_step.py --model $MODEL --use-together-api  > $LOG_DIR/mft_two_step.log 2>&1
# python cot_bottleneck.py --model $MODEL --use-together-api  > $LOG_DIR/cot.log 2>&1




