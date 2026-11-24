#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Parallel version of launch.sh: runs up to JOBS configs at the same time on
# one machine (e.g. several small CIFAR-10 models sharing an Apple-Silicon GPU).
# Written for the bash 3.2 that ships with macOS (no `wait -n`, no assoc arrays).
#
# Usage:
#   bash launch_parallel.sh TASK DIVISION GPU JOBS [WORKERS]
#     TASK      mlp_cifar10 | mlp_cifar10_bio | cnn_imagenet | resnet34_imagenet |
#               resnet34sep_imagenet | resnet50_imagenet | efficient_imagenet
#     DIVISION  best | all | g52 | repro | ...  (folder under configs/baselines/<dataset>/<task>/)
#     GPU       1 | 8   (picks the 1-GPU or 8-GPU ImageNet configs)
#     JOBS      number of configs to train at once      (default 4)
#     WORKERS   data-loader workers per job             (default 2)
#
#   Extra config overrides can be passed through the EXTRA variable, e.g.
#     EXTRA="OPTIM.MAX_EPOCH 2 RGRAPH.SEED_TRAIN_END 2" bash launch_parallel.sh mlp_cifar10 all 1 4
#
#   Several NVIDIA GPUs: list them in GPUS; each job goes to the GPU with the fewest running
#   jobs (CUDA_VISIBLE_DEVICES), e.g. 3 jobs per GPU on two GPUs:
#     GPUS="0 1" bash launch_parallel.sh mlp_cifar10 g52 1 6 2
#   Without GPUS every job uses the default device (the only GPU on a Mac).
#
# By default every (config, training seed) pair is a separate job, so even
# `best` (2 configs x 5 seeds = 10 jobs) runs in parallel. Logs go to
#   checkpoint/<dataset>/<task>/<division>/<config>/run_seed<N>.log
# Set SPLIT_SEEDS=0 to run each config's seeds sequentially in one job (run.log).
# Split one division between machines with SHARD=i/n: the i-th of n contiguous blocks of configs (i = 1..n),
# e.g. SHARD=1/2 on one machine and SHARD=2/2 on another; copy the checkpoint folders together afterwards.
# Re-running the same command skips seeds that already finished and restarts
# unfinished ones cleanly (results are identical to launch.sh).

TASK=${1:-mlp_cifar10}
DIVISION=${2:-best}
GPU=${3:-1}
JOBS=${4:-4}
WORKERS=${5:-2}
EXTRA=${EXTRA:-}
GPUS=${GPUS:-}

if [ "$TASK" = "mlp_cifar10" ] || [ "$TASK" = "mlp_cifar10_bio" ]; then
    DATASET='cifar10'; PREFIX='mlp_bs128_1gpu_layer'      # layer3: 5-layer MLP, layer6: 8-layer MLP
elif [ "$TASK" = "cnn_imagenet" ]; then
    DATASET='imagenet'
    if [ "$GPU" = 1 ]; then PREFIX='cnn6_bs32_1gpu_64d'; else PREFIX='cnn6_bs256_8gpu_64d'; fi
elif [ "$TASK" = "resnet34_imagenet" ] || [ "$TASK" = "resnet34sep_imagenet" ]; then
    DATASET='imagenet'
    if [ "$GPU" = 1 ]; then PREFIX='R-34_bs32_1gpu'; else PREFIX='R-34_bs256_8gpu'; fi
elif [ "$TASK" = "resnet50_imagenet" ]; then
    DATASET='imagenet'
    if [ "$GPU" = 1 ]; then PREFIX='R-50_bs32_1gpu'; else PREFIX='R-50_bs256_8gpu'; fi
elif [ "$TASK" = "efficient_imagenet" ]; then
    DATASET='imagenet'
    if [ "$GPU" = 1 ]; then PREFIX='EN-B0_bs64_1gpu_nms'; else PREFIX='EN-B0_bs512_8gpu_nms'; fi
else
    echo "Unknown task: $TASK"; exit 1
fi

CFG_DIR=configs/baselines/${DATASET}/${TASK}/${DIVISION}
OUT_ROOT=checkpoint/${DATASET}/${TASK}/${DIVISION}

# Collect matching configs
CONFIGS=()
for CONFIG in "$CFG_DIR"/*.yaml; do
    case "$CONFIG" in *"$PREFIX"*) CONFIGS+=("$CONFIG") ;; esac
done
TOTAL=${#CONFIGS[@]}
if [ "$TOTAL" -eq 0 ]; then echo "No configs found in $CFG_DIR matching $PREFIX"; exit 1; fi
if [ -n "${SHARD:-}" ]; then
    SI=${SHARD%/*}; SN=${SHARD#*/}
    A=$(( (SI - 1) * TOTAL / SN )); B=$(( SI * TOTAL / SN ))
    CONFIGS=("${CONFIGS[@]:$A:$((B - A))}")
    echo "[$(date '+%H:%M:%S')] shard $SHARD: configs $((A + 1))-$B of $TOTAL"
    TOTAL=${#CONFIGS[@]}
fi
echo "[$(date '+%H:%M:%S')] running $JOBS jobs at a time ($WORKERS loader workers each)$( [ -n "$GPUS" ] && echo " on GPUs: $GPUS")"

PIDS=()
NAMES=()
LOGS=()
JOB_GPU=()
REPORTED=()
FAILED=0
# Ctrl-C stops every running job
stop_all() {
    echo; echo "Stopping all jobs..."
    # stop data-loader workers first, then the training processes
    for p in "${PIDS[@]}"; do pkill -TERM -P "$p" 2>/dev/null; kill "$p" 2>/dev/null; done
    for p in "${PIDS[@]}"; do while kill -0 "$p" 2>/dev/null; do sleep 0.5; done; done
    echo "All jobs stopped."
    exit 130
}
trap stop_all INT TERM

running_jobs() { jobs -rp | wc -l | tr -d ' '; }
is_running() { case " $(jobs -rp | tr '\n' ' ') " in *" $1 "*) return 0 ;; esac; return 1; }

# report jobs that have finished since the last call (as they finish, not only at the end)
reap() {
    local k
    for k in "${!PIDS[@]}"; do
        [ "${REPORTED[$k]}" = 1 ] && continue
        is_running "${PIDS[$k]}" && continue
        if wait "${PIDS[$k]}"; then
            echo "[$(date '+%H:%M:%S')] done   ${NAMES[$k]}"
        else
            echo "[$(date '+%H:%M:%S')] FAILED ${NAMES[$k]}  (see ${LOGS[$k]})"
            FAILED=$((FAILED + 1))
        fi
        REPORTED[$k]=1
    done
}

# GPU with the fewest running jobs (only used when GPUS is set)
pick_gpu() {
    local g k n best="" bestn=999999
    for g in $GPUS; do
        n=0
        for k in "${!PIDS[@]}"; do
            [ "${JOB_GPU[$k]}" = "$g" ] && [ "${REPORTED[$k]}" != 1 ] && is_running "${PIDS[$k]}" && n=$((n + 1))
        done
        if [ "$n" -lt "$bestn" ]; then best=$g; bestn=$n; fi
    done
    echo "$best"
}

# value of "KEY: v" in a yaml config, overridden by "SECTION.KEY v" in EXTRA
cfg_val() {  # $1=config  $2=SECTION.KEY
    local key=${2##*.} v
    v=$(echo " $EXTRA " | sed -n "s/.* $2 \([^ ]*\) .*/\1/p")
    [ -z "$v" ] && v=$(grep -E "^[[:space:]]+$key:" "$1" | head -1 | awk '{print $2}')
    echo "$v"
}

# Build the job list: one job per (config, seed) unless SPLIT_SEEDS=0
JOB_CFG=(); JOB_SEED=()
for CONFIG in "${CONFIGS[@]}"; do
    if [ "${SPLIT_SEEDS:-1}" = 1 ]; then
        S0=$(cfg_val "$CONFIG" RGRAPH.SEED_TRAIN_START); S1=$(cfg_val "$CONFIG" RGRAPH.SEED_TRAIN_END)
        S0=${S0:-1}; S1=${S1:-2}
        S=$S0
        while [ "$S" -lt "$S1" ]; do JOB_CFG+=("$CONFIG"); JOB_SEED+=("$S"); S=$((S + 1)); done
    else
        JOB_CFG+=("$CONFIG"); JOB_SEED+=("all")
    fi
done
NJOBS=${#JOB_CFG[@]}
echo "[$(date '+%H:%M:%S')] $NJOBS jobs ($TOTAL configs$( [ "${SPLIT_SEEDS:-1}" = 1 ] && echo ' x seeds'))"

SKIPPED=0
for k in "${!JOB_CFG[@]}"; do
    CONFIG=${JOB_CFG[$k]}; SEED=${JOB_SEED[$k]}
    NAME=$(basename "$CONFIG" .yaml)
    OUT_DIR=${OUT_ROOT}/${NAME}
    if [ "$SEED" = all ]; then
        SEED_ARGS=""; LOG="$OUT_DIR/run.log"; TAG="$NAME"
    else
        # skip seeds that already finished (seed is the last column of the results file)
        RES="$OUT_DIR/results_epoch$(cfg_val "$CONFIG" OPTIM.MAX_EPOCH).txt"
        if [ -f "$RES" ] && awk -F', *' -v s="$SEED" '$NF==s {f=1} END {exit !f}' "$RES"; then
            SKIPPED=$((SKIPPED + 1)); continue
        fi
        SEED_ARGS="RGRAPH.SEED_TRAIN_START $SEED RGRAPH.SEED_TRAIN_END $((SEED + 1))"
        LOG="$OUT_DIR/run_seed${SEED}.log"; TAG="$NAME [seed $SEED]"
    fi
    # wait for a free slot
    while [ "$(running_jobs)" -ge "$JOBS" ]; do sleep 2; reap; done
    reap
    mkdir -p "$OUT_DIR"
    G=""
    [ -n "$GPUS" ] && G=$(pick_gpu)
    echo "[$(date '+%H:%M:%S')] start ($((k + 1))/$NJOBS) $TAG$( [ -n "$G" ] && echo "  [GPU $G]")"
    if [ -n "$G" ]; then
        CUDA_VISIBLE_DEVICES=$G python tools/train_net.py --cfg "$CONFIG" TRAIN.AUTO_RESUME False OUT_DIR "$OUT_DIR" \
            BN.USE_PRECISE_STATS True LOG_DEST stdout DATA_LOADER.NUM_WORKERS "$WORKERS" $EXTRA $SEED_ARGS \
            > "$LOG" 2>&1 &
    else
        python tools/train_net.py --cfg "$CONFIG" TRAIN.AUTO_RESUME False OUT_DIR "$OUT_DIR" \
            BN.USE_PRECISE_STATS True LOG_DEST stdout DATA_LOADER.NUM_WORKERS "$WORKERS" $EXTRA $SEED_ARGS \
            > "$LOG" 2>&1 &
    fi
    PIDS+=($!)
    NAMES+=("$TAG")
    LOGS+=("$LOG")
    JOB_GPU+=("$G")
    REPORTED+=(0)
done
[ "$SKIPPED" -gt 0 ] && echo "[$(date '+%H:%M:%S')] skipped $SKIPPED already-finished jobs"

# Wait for everything, reporting jobs as they finish
while [ "$(running_jobs)" -gt 0 ]; do sleep 2; reap; done
reap
echo "[$(date '+%H:%M:%S')] finished: $((${#PIDS[@]} - FAILED))/${#PIDS[@]} succeeded"
[ "$FAILED" -eq 0 ]
