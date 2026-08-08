#!/bin/bash
set -euo pipefail

# Run from the directory containing cp2k.slurm and runs/.
# Arguments:
#   1: manifest path (default runs/inas_incl3_dft_k3/manifest.tsv)
#   2: isomers per array task (default 1)
#   3: maximum concurrent array tasks (default 8)

CURRENT_DIR=$(pwd)
MANIFEST=${1:-$CURRENT_DIR/runs/inas_incl3_dft_k3/manifest.tsv}
BATCH_SIZE=${2:-1}
MAX_CONCURRENT=${3:-8}
SLURM_SCRIPT=$CURRENT_DIR/cp2k.slurm

if [[ ! -f "$SLURM_SCRIPT" ]]; then
  echo "cp2k.slurm not found in current directory: $CURRENT_DIR" >&2
  exit 2
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST" >&2
  exit 2
fi
if ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH_SIZE must be a positive integer" >&2
  exit 2
fi
if ! [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_CONCURRENT must be a positive integer" >&2
  exit 2
fi

MANIFEST=$(realpath "$MANIFEST")
TOTAL=$(( $(wc -l < "$MANIFEST") - 1 ))
if (( TOTAL < 1 )); then
  echo "Manifest contains no structures: $MANIFEST" >&2
  exit 2
fi

TASKS=$(( (TOTAL + BATCH_SIZE - 1) / BATCH_SIZE ))
LAST=$((TASKS - 1))
echo "Current directory: $CURRENT_DIR"
echo "Manifest: $MANIFEST"
echo "Structures: $TOTAL"
echo "Array tasks: $TASKS"
echo "Isomers per task: $BATCH_SIZE"
echo "Maximum concurrent tasks: $MAX_CONCURRENT"

sbatch \
  --array="0-${LAST}%${MAX_CONCURRENT}" \
  --export="ALL,MANIFEST=$MANIFEST,BATCH_SIZE=$BATCH_SIZE" \
  "$SLURM_SCRIPT"
