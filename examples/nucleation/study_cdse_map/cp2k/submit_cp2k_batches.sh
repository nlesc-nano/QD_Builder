#!/bin/bash
set -euo pipefail

if (( $# < 1 || $# > 3 )); then
  echo "Usage: $0 MANIFEST [BATCH_SIZE=1] [MAX_CONCURRENT]" >&2
  exit 2
fi

MANIFEST=$(realpath "$1")
BATCH_SIZE=${2:-1}
MAX_CONCURRENT=${3:-}
if ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH_SIZE must be a positive integer" >&2
  exit 2
fi
if [[ -n "$MAX_CONCURRENT" ]] && ! [[ "$MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_CONCURRENT must be a positive integer" >&2
  exit 2
fi

TOTAL=$(( $(wc -l < "$MANIFEST") - 1 ))
if (( TOTAL < 1 )); then
  echo "Manifest contains no structures: $MANIFEST" >&2
  exit 2
fi
TASKS=$(( (TOTAL + BATCH_SIZE - 1) / BATCH_SIZE ))
LAST=$((TASKS - 1))
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

echo "Submitting $TOTAL structures as $TASKS array tasks"
if [[ -n "$MAX_CONCURRENT" ]]; then
  ARRAY_SPEC="0-${LAST}%${MAX_CONCURRENT}"
  echo "Batch size: $BATCH_SIZE; maximum concurrent tasks: $MAX_CONCURRENT"
else
  ARRAY_SPEC="0-${LAST}"
  echo "Batch size: $BATCH_SIZE; concurrency controlled by the HPC scheduler"
fi
sbatch \
  --array="$ARRAY_SPEC" \
  --export="ALL,MANIFEST=$MANIFEST,BATCH_SIZE=$BATCH_SIZE" \
  "$SCRIPT_DIR/cp2k.slurm"
