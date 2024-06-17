start_lr=$1
num_epochs=$2
log_name=$3
model_prefix_name=$4

DATA_DIR_TRAIN="/data/lxiao/Sig_Bkg_EvenT/train"
DATA_DIR_VAL="/data/lxiao/Sig_Bkg_EvenT/val"
DATA_DIR_TEST="/data/lxiao/Sig_Bkg_EvenT/test"
WEAVER_DIR="/home/lxiao/weaver-main"

data_config="${WEAVER_DIR}/test/JetClass_full.yaml"
network_config="${WEAVER_DIR}/test/example_ParticleTransformer.py"
model_prefix="${WEAVER_DIR}/models/${model_prefix_name}"
log_path="${WEAVER_DIR}/logs/${log_name}"

cd  /home/lxiao/weaver-main
python3 weaver/train.py
    --data-train \
    "Sig_HH:${DATA_DIR_TRAIN}/sig_hh/Sig_HH*.root" \
    "Bkg_ZZ:${DATA_DIR_TRAIN}/bkg_zz/Bkg_ZZ*.root" \
    "Bkg_ZH:${DATA_DIR_TRAIN}/bkg_zh/Bkg_ZH*.root" \
    "Bkg_TT:${DATA_DIR_TRAIN}/bkg_tt/Bkg_TT*.root" \
    "Bkg_TTH:${DATA_DIR_TRAIN}/bkg_tth/Bkg_TTH*.root" \
    "Bkg_TT2B:${DATA_DIR_TRAIN}/bkg_tt2b/Bkg_TT2B*.root" \
    "Bkg_H2B:${DATA_DIR_TRAIN}/bkg_h2b/Bkg_H2B*.root" \
    "Bkg_2B2J:${DATA_DIR_TRAIN}/bkg_2b2j/Bkg_2B2J*.root" \
    "Bkg_4B:${DATA_DIR_TRAIN}/bkg_4b/Bkg_4B*.root" \
    --data-val \
    "Sig_HH:${DATA_DIR_VAL}/sig_hh/Sig_HH*.root" \
    "Bkg_ZZ:${DATA_DIR_VAL}/bkg_zz/Bkg_ZZ*.root" \
    "Bkg_ZH:${DATA_DIR_VAL}/bkg_zh/Bkg_ZH*.root" \
    "Bkg_TT:${DATA_DIR_VAL}/bkg_tt/Bkg_TT*.root" \
    "Bkg_TTH:${DATA_DIR_VAL}/bkg_tth/Bkg_TTH*.root" \
    "Bkg_TT2B:${DATA_DIR_VAL}/bkg_tt2b/Bkg_TT2B*.root" \
    "Bkg_H2B:${DATA_DIR_VAL}/bkg_h2b/Bkg_H2B*.root" \
    "Bkg_2B2J:${DATA_DIR_VAL}/bkg_2b2j/Bkg_2B2J*.root" \
    "Bkg_4B:${DATA_DIR_VAL}/bkg_4b/Bkg_4B*.root" \
    --data-test \
    "Sig_HH:${DATA_DIR_TEST}/sig_hh/Sig_HH*.root" \
    "Bkg_ZZ:${DATA_DIR_TEST}/bkg_zz/Bkg_ZZ*.root" \
    "Bkg_ZH:${DATA_DIR_TEST}/bkg_zh/Bkg_ZH*.root" \
    "Bkg_TT:${DATA_DIR_TEST}/bkg_tt/Bkg_TT*.root" \
    "Bkg_TTH:${DATA_DIR_TEST}/bkg_tth/Bkg_TTH*.root" \
    "Bkg_TT2B:${DATA_DIR_TEST}/bkg_tt2b/Bkg_TT2B*.root" \
    "Bkg_H2B:${DATA_DIR_TEST}/bkg_h2b/Bkg_H2B*.root" \
    "Bkg_2B2J:${DATA_DIR_TEST}/bkg_2b2j/Bkg_2B2J*.root" \
    "Bkg_4B:${DATA_DIR_TEST}/bkg_4b/Bkg_4B*.root" \
    --data-config $data_config \
    --network-config $network_config \
    --model-prefix $model_prefix \
    --gpus 0 \
    --batch-size 128 \
    --start-lr $start_lr \
    --num-epochs $num_epochs \
    --optimizer ranger \
    --log $log_path \


