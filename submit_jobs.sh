#!/bin/bash
set -euo pipefail

# Run this script from the directory containing cp2k.slurm and runs/.
# Optional arguments:
#   1: manifest path    (default runs/cdse_map/dft_all/manifest.tsv)
#   2: batch size       (default 1 isomer per array task)
#   3: max concurrent   (optional; omitted = let HPC scheduler decide)
CURRENT_DIR=$(pwd)
MANIFEST=${1:-$CURRENT_DIR/runs/cdse_map/dft_all/manifest.tsv}
BATCH_SIZE=${2:-1}
MAX_CONCURRENT=${3:-}
SLURM_SCRIPT=$CURRENT_DIR/cp2k.slurm

if [[ ! -f "$SLURM_SCRIPT" ]]; then
  echo "cp2k.slurm not found in the current directory: $CURRENT_DIR" >&2
  exit 2
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST" >&2
  echo "Expected the prepared DFT tree under runs/cdse_map/dft_all/." >&2
  exit 2
fi
if ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH_SIZE must be a positive integer" >&2
  exit 2
fi
if [[ -n "$MAX_CONCURRENT" ]] && ! [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
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
if [[ -n "$MAX_CONCURRENT" ]]; then
  ARRAY_SPEC="0-${LAST}%${MAX_CONCURRENT}"
  echo "Maximum concurrent tasks: $MAX_CONCURRENT"
else
  ARRAY_SPEC="0-${LAST}"
  echo "Maximum concurrent tasks: controlled by the HPC scheduler"
fi

sbatch \
  --array="$ARRAY_SPEC" \
  --export="ALL,MANIFEST=$MANIFEST,BATCH_SIZE=$BATCH_SIZE" \
  "$SLURM_SCRIPT"
