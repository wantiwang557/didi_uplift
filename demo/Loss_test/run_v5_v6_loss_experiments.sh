#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   cd /nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/demo/Loss_test
#   bash run_v5_v6_loss_experiments.sh v5 ziln
#   bash run_v5_v6_loss_experiments.sh v5 tweedie
#   bash run_v5_v6_loss_experiments.sh v5 mil
#   bash run_v5_v6_loss_experiments.sh v5 mil_ziln
#   bash run_v5_v6_loss_experiments.sh v6 ziln
#   bash run_v5_v6_loss_experiments.sh v6 tweedie
#   bash run_v5_v6_loss_experiments.sh v6 mil
#   bash run_v5_v6_loss_experiments.sh v6 mil_ziln
#
# Notes:
#   1. v5: train 0511-0821, valid 0822-0827, test 0828-0910.
#   2. v6: train 0716-0821, valid 0822-0827, test 0828-0910.
#   3. valid/test use train feature conf and parser; model selection uses valid, final effect uses test.
#   4. Explicitly keep token_size=15, DENSE_FEA_SIZE=3500, PADDING_SIZE=473.
#   5. GPU defaults to 0. Override with GPU_ID=1 bash run_v5_v6_loss_experiments.sh v5 ziln.

DATASET="${1:?dataset must be v5 or v6}"
FAMILY="${2:?family must be ziln, tweedie, mil, or mil_ziln}"
GPU_ID="${GPU_ID:-0}"

BASE_TFR="/nfs/dataset-ofs-385-143496/yangliu/yangliu_data/tfrecord_cache/euen_010203/wwt"
BASE_MODEL="hdfs://difed/user/prod_growstrategy/wwt/demo/multi_model/Loss_test"

case "${DATASET}" in
  v5)
    TRAIN_TFR="${BASE_TFR}/v5/train_0511_0821/part-*"
    VALID_TFR="${BASE_TFR}/v5/valid_0822_0827/part-*"
    TEST_TFR="${BASE_TFR}/v5/test_0828_0910/part-*"
    TRAIN_START_DT="20260511"
    TRAIN_END_DT="20260821"
    TEST_DT="20260828"
    PERIOD_TAG="v5_train_0511_0821_valid_0822_0827_test_0828_0910"
    ;;
  v6)
    TRAIN_TFR="${BASE_TFR}/v6/train_0716_0821/part-*"
    VALID_TFR="${BASE_TFR}/v6/valid_0822_0827/part-*"
    TEST_TFR="${BASE_TFR}/v6/test_0828_0910/part-*"
    TRAIN_START_DT="20260716"
    TRAIN_END_DT="20260821"
    TEST_DT="20260828"
    PERIOD_TAG="v6_train_0716_0821_valid_0822_0827_test_0828_0910"
    ;;
  *)
    echo "Unknown dataset: ${DATASET}. Use v5 or v6." >&2
    exit 1
    ;;
esac

COMMON_ARGS=(
  --train_tfr_path "${TRAIN_TFR}"
  --valid_tfr_path "${VALID_TFR}"
  --test_tfr_path "${TEST_TFR}"
  --train_start_dt "${TRAIN_START_DT}"
  --train_end_dt "${TRAIN_END_DT}"
  --test_dt "${TEST_DT}"
  --log_interval 100
  --shuffle_seed 0
  --parallel 4
  --DENSE_FEA_SIZE 3500
  --PADDING_SIZE 473
  --token_size 15
  --gpu "${GPU_ID}"
)

run_ziln() {
  mkdir -p ./run_logs/ziln
  for ver in 01 02 03 04; do
    script="euen_4t_ziln_${ver}_batch.py"
    version="euen_4t_ziln_${ver}_batch_${PERIOD_TAG}"
    log="./run_logs/ziln/${version}.log"
    nohup python "${script}" \
      "${COMMON_ARGS[@]}" \
      --model_save_path "${BASE_MODEL}/euen_4t_ziln_${ver}_batch/${PERIOD_TAG}" \
      --version "${version}" \
      > "${log}" 2>&1 &
    echo "started ${version}, log=${log}"
  done
}

run_tweedie() {
  mkdir -p ./run_logs/tweedie
  for ver in 01 02 03; do
    script="euen_4t_tweedie_${ver}.py"
    version="euen_4t_tweedie_${ver}_${PERIOD_TAG}"
    log="./run_logs/tweedie/${version}.log"
    nohup python "${script}" \
      "${COMMON_ARGS[@]}" \
      --tweedie_p_list "1.1,1.3,1.5,1.7,1.9" \
      --model_save_path "${BASE_MODEL}/euen_4t_tweedie_${ver}/${PERIOD_TAG}" \
      --version "${version}" \
      > "${log}" 2>&1 &
    echo "started ${version}, log=${log}"
  done
}

run_mil() {
  mkdir -p ./run_logs/mil
  version="euen_4t_MIL_01_${PERIOD_TAG}"
  log="./run_logs/mil/${version}.log"
  nohup python euen_4t_MIL_01.py \
    "${COMMON_ARGS[@]}" \
    --mil_bag_size_list "64,128" \
    --mil_alpha_list "0.000001" \
    --model_save_path "${BASE_MODEL}/euen_4t_MIL_01/${PERIOD_TAG}" \
    --version "${version}" \
    > "${log}" 2>&1 &
  echo "started ${version}, log=${log}"
}

run_mil_ziln() {
  mkdir -p ./run_logs/mil
  for ver in 01 03 04; do
    script="euen_4t_mil_ziln_${ver}.py"
    version="euen_4t_mil_ziln_${ver}_${PERIOD_TAG}"
    log="./run_logs/mil/${version}.log"
    nohup python "${script}" \
      "${COMMON_ARGS[@]}" \
      --mil_bag_size 128 \
      --alpha_mil 0.000001 \
      --model_save_path "${BASE_MODEL}/euen_4t_mil_ziln_${ver}/${PERIOD_TAG}" \
      --version "${version}" \
      > "${log}" 2>&1 &
    echo "started ${version}, log=${log}"
  done
}

case "${FAMILY}" in
  ziln) run_ziln ;;
  tweedie) run_tweedie ;;
  mil) run_mil ;;
  mil_ziln) run_mil_ziln ;;
  *)
    echo "Unknown family: ${FAMILY}. Use ziln, tweedie, mil, or mil_ziln." >&2
    exit 1
    ;;
esac
