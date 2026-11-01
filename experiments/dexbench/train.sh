#!/usr/bin/env bash
# DexBench GA wrist-camera finetuning of FastWAM Joint on one node, through the upstream ZeRO-2 launcher.
# Keeps the global batch at 256: batch_size (per GPU) x NUM_GPUS x gradient_accumulation_steps.
# Extra arguments are Hydra overrides, e.g. 'data.train.dataset_dirs=[/path/a,/path/b]'.
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GLOBAL_BATCH=256
TASK="${TASK:-dexbench_ga_wrist_joint_3cam_384_1e-4}"

if (( GLOBAL_BATCH % (BATCH_SIZE * NUM_GPUS) != 0 )); then
    echo "BATCH_SIZE (${BATCH_SIZE}) x NUM_GPUS (${NUM_GPUS}) must divide ${GLOBAL_BATCH}" >&2
    exit 1
fi
ACCUMULATION=$(( GLOBAL_BATCH / (BATCH_SIZE * NUM_GPUS) ))

exec bash scripts/train_zero2.sh "${NUM_GPUS}" "task=${TASK}" \
    "batch_size=${BATCH_SIZE}" "gradient_accumulation_steps=${ACCUMULATION}" "$@"
