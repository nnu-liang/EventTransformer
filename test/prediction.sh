
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_root_path> <output_root_path>"
    exit 1
fi

INPUT_ROOT_PATH=$1
OUTPUT_ROOT_PATH=$2

WEAVER_DIR="/home/lxiao/weaver-main"

data_config="${WEAVER_DIR}/test/JetClass_full.yaml"
network_config="${WEAVER_DIR}/test/example_ParticleTransformer.py"
model_prefix="${WEAVER_DIR}/models/train3_best_epoch_state.pt"

cd $WEAVER_DIR

python3 weaver/train.py \
    --predict \
    --data-test ${INPUT_ROOT_PATH} \
    --data-config $data_config \
    --network-config $network_config \
    --model-prefix $model_prefix \
    --gpus 0 \
    --batch-size 512 \
    --predict-output $OUTPUT_ROOT_PATH



