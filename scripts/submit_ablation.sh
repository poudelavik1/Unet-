#!/bin/bash
# Optimiser and loss study: one training run per combination of OPTIMIZERS and LOSSES, one job
# after another, then the comparison figures. ASL has removed a second job of the same user from
# the queue while the first was running, so a job is submitted only when the queue is empty.
# Start it in the background on the login node; it survives the SSH session:
#
#   nohup bash scripts/submit_ablation.sh "$HOME/dataset_20261011/split_80_10_10" > logs/ablation.log 2>&1 &
#
# A run that has finished (test_metrics.json) is skipped and a run with a last.pt is resumed, so
# the same command continues the study after an interruption. Every other variable of
# scripts/submit.sh (ENCODER, EPOCHS, AUG_PRESET, TRAIN_ARGS, ...) applies to every run.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -ne 1 ]]; then
    echo "Usage: bash scripts/submit_ablation.sh /absolute/dataset/path" >&2
    exit 2
fi
DATASET="$1"
OPTIMIZERS="${OPTIMIZERS:-adamw adam}"
# One loss per word: a term alone, "hairline" (BCE + Tversky + clDice), or terms joined by "+".
LOSSES="${LOSSES:-hairline bce focal dice tversky}"
STUDY_DIR="$(realpath -m "${STUDY_DIR:-$ROOT/runs/ablation}")"
ENV_DIR="${ENV_DIR:-$HOME/.conda/envs/${ENV_NAME:-swinunet}}"
mkdir -p "$STUDY_DIR" "$ROOT/logs"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
queue_empty() {
    local jobs
    jobs=$(squeue -h -u "$USER") && [[ -z "$jobs" ]]  # a failed squeue does not count as empty
}
wait_for_queue() {
    until queue_empty; do sleep 120; done
}

echo "$(stamp) study: optimisers [$OPTIMIZERS] x losses [$LOSSES] in $STUDY_DIR"
for optimizer in $OPTIMIZERS; do
    for loss in $LOSSES; do
        run="$STUDY_DIR/${optimizer}_${loss//+/-}"
        if [[ -f "$run/test_metrics.json" ]]; then
            echo "$(stamp) finished already: $run"
            continue
        fi
        wait_for_queue
        resume=0
        if [[ -f "$run/last.pt" ]]; then
            resume=1
        elif [[ -e "$run" ]]; then
            echo "$(stamp) skipped: $run exists without last.pt or test_metrics.json (did the job fail?);" \
                 "remove the folder to train it again"
            continue
        fi
        echo "$(stamp) submitting optimizer=$optimizer loss='${loss//+/ }' resume=$resume -> $run"
        OPTIMIZER="$optimizer" LOSS="${loss//+/ }" RESUME="$resume" \
            bash "$ROOT/scripts/submit.sh" "$DATASET" "$run" || echo "$(stamp) submission failed: $run"
        sleep 120  # let the job appear in the queue before it is checked again
    done
done
wait_for_queue
echo "$(stamp) every run has ended; drawing the comparison"
LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}" MPLBACKEND=Agg \
    "$ENV_DIR/bin/python" "$ROOT/scripts/compare_runs.py" "$STUDY_DIR" --output "$STUDY_DIR/comparison"
echo "$(stamp) done"
