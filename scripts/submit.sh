#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: bash scripts/submit.sh /absolute/dataset/path [output-directory]" >&2
    exit 2
fi
DATASET="$(realpath -e "$1")" || { echo "Dataset path does not exist: $1" >&2; exit 1; }
export DATASET
for split in train val test; do
    for kind in images masks; do
        test -d "$DATASET/$split/$kind" || { echo "Missing $DATASET/$split/$kind" >&2; exit 1; }
    done
done
if [[ -n "${SELECT_PATTERN:-}" ]]; then
    # The trainer silently falls back to all sources when nothing matches, so check before queueing.
    matched=$(ls "$DATASET/val/images" | grep -cE -- "$SELECT_PATTERN" || true)
    [[ "$matched" -gt 0 ]] || { echo "SELECT_PATTERN matches no validation image: $SELECT_PATTERN" >&2; exit 1; }
    echo "Target: $matched validation images match $SELECT_PATTERN"
fi
if [[ -n "${INIT_WEIGHTS:-}" ]]; then
    # Resolve before the cd below, and fail here rather than after the queue wait.
    INIT_WEIGHTS="$(realpath -e "$INIT_WEIGHTS")" || { echo "INIT_WEIGHTS does not exist" >&2; exit 1; }
    export INIT_WEIGHTS
fi
# The GSD table is for the data pipeline only: it keeps the mixing augmentations within one
# GSD and gives the per-GSD test metrics. The network does not use it.
GSD_CSV="$(realpath -e "${GSD_CSV:-$ROOT/configs/gsd.csv}")" || { echo "GSD_CSV does not exist" >&2; exit 1; }
export GSD_CSV
while IFS= read -r row; do
    pattern="${row%,*}"
    count=$(ls "$DATASET/train/images" | grep -cE -- "$pattern" || true)
    printf 'GSD %s mm/px: %s training images match %s\n' "${row##*,}" "$count" "$pattern"
done < <(tail -n +2 "$GSD_CSV" | tr -d '\r' | grep -vE '^[[:space:]]*(#|$)')
ENCODER="${ENCODER:-tu-convnext_tiny}"
export ENCODER
echo "Encoder: $ENCODER"
if [[ "${RESUME:-0}" != 1 && "${EVALUATE_ONLY:-0}" != 1 && -z "${INIT_WEIGHTS:-}" ]]; then
    # Compute nodes may lack internet, so the pretrained encoder must already be in the cache.
    # Checked as the job will load it: with the environment's Python and the Hugging Face Hub offline.
    ENV_DIR="${ENV_DIR:-$HOME/.conda/envs/${ENV_NAME:-swinunet}}"
    test -x "$ENV_DIR/bin/python" || {
        echo "Conda environment not found: $ENV_DIR. Run: bash scripts/setup_conda.sh" >&2; exit 1; }
    LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}" HF_HUB_OFFLINE=1 \
        "$ENV_DIR/bin/python" "$ROOT/scripts/encoder_weights.py" "$ENCODER" || {
        echo "Pretrained $ENCODER weights are not cached. Run: ENCODER=$ENCODER bash scripts/setup_conda.sh" >&2
        exit 1
    }
fi
cd "$ROOT"
mkdir -p logs runs
export OUTPUT_DIR="$(realpath -m "${2:-$ROOT/runs/$(date +%Y%m%d_%H%M%S)_$$}")"
if [[ "${EVALUATE_ONLY:-0}" == 1 ]]; then
    # Scores the best.pt of an earlier run again; the run's own files stay as they are.
    [[ $# -eq 2 ]] || { echo "EVALUATE_ONLY=1 needs the run directory as the second argument" >&2; exit 1; }
    test -f "$OUTPUT_DIR/best.pt" || { echo "No best.pt to evaluate in $OUTPUT_DIR" >&2; exit 1; }
elif [[ "${RESUME:-0}" == 1 ]]; then
    test -f "$OUTPUT_DIR/last.pt" || { echo "No last.pt to resume" >&2; exit 1; }
elif [[ -e "$OUTPUT_DIR" ]]; then
    echo "Output already exists. Choose a new directory or use RESUME=1." >&2
    exit 1
fi
# mkdir belongs to the allocated job, so a failed submission leaves no results folder.
submission=(--parsable --export=ALL)
[[ -z "${SLURM_ACCOUNT:-}" ]] || submission+=(--account="$SLURM_ACCOUNT")
[[ -z "${SLURM_PARTITION:-}" ]] || submission+=(--partition="$SLURM_PARTITION")
raw_job=$(sbatch "${submission[@]}" slurm/train_unetpp.sbatch)
job_id="${raw_job%%;*}"
printf 'Job: %s\nResults: %s\n' "$job_id" "$OUTPUT_DIR"
printf 'Watch log: tail -F logs/unetpp_%s.out\n' "$job_id"
printf 'Watch metrics: python scripts/monitor_epochs.py "%s" --watch\n' "$OUTPUT_DIR"
