#!/bin/bash
#SBATCH -J CdSe_mol_DFT
#SBATCH -t 24:00:00
#SBATCH -n 24
#SBATCH --mem-per-cpu=2g
#SBATCH --qos=regular
#SBATCH --array=0-0

set -uo pipefail
module purge
module load CP2K/2024.3-foss-2023a
export OMP_NUM_THREADS=1

MANIFEST=${MANIFEST:-manifest.tsv}
BATCH_SIZE=${BATCH_SIZE:-1}
BATCH_ID=${SLURM_ARRAY_TASK_ID:-0}

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST" >&2
  exit 2
fi
if ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH_SIZE must be a positive integer" >&2
  exit 2
fi

MANIFEST=$(realpath "$MANIFEST")
RUN_ROOT=$(dirname "$MANIFEST")
mapfile -t ROWS < <(tail -n +2 "$MANIFEST")
TOTAL=${#ROWS[@]}
FIRST=$((BATCH_ID * BATCH_SIZE))
LAST=$((FIRST + BATCH_SIZE - 1))
if (( FIRST >= TOTAL )); then
  echo "Array task $BATCH_ID starts beyond manifest; nothing to do"
  exit 0
fi
if (( LAST >= TOTAL )); then
  LAST=$((TOTAL - 1))
fi

echo "Array task $BATCH_ID: rows $FIRST-$LAST of $TOTAL"
FAILED=0
for ((row_index=FIRST; row_index<=LAST; row_index++)); do
  IFS=$'\t' read -r INDEX K P STRUCTURE_ID BOX RUN_DIR SOURCE_XYZ <<< "${ROWS[$row_index]}"
  CALC_DIR="$RUN_ROOT/$RUN_DIR"
  echo "[$INDEX/$TOTAL] k=$K p=$P id=$STRUCTURE_ID box=${BOX} A"
  echo "  source: $SOURCE_XYZ"
  echo "  run:    $CALC_DIR"

  if [[ ! -f "$CALC_DIR/cp2k_job.in" || ! -f "$CALC_DIR/start.xyz" ]]; then
    echo "Missing cp2k_job.in or start.xyz under $CALC_DIR" >&2
    FAILED=1
    continue
  fi
  if [[ -f "$CALC_DIR/cp2k_job.out" ]] && grep -q "PROGRAM ENDED" "$CALC_DIR/cp2k_job.out"; then
    touch "$CALC_DIR/.done"
    echo "  already completed; skipping"
    continue
  fi

  rm -f "$CALC_DIR/.done" "$CALC_DIR/.failed" "$CALC_DIR/.running"
  touch "$CALC_DIR/.running"
  if (cd "$CALC_DIR" && srun cp2k.popt -i cp2k_job.in -o cp2k_job.out); then
    if grep -q "PROGRAM ENDED" "$CALC_DIR/cp2k_job.out"; then
      touch "$CALC_DIR/.done"
      echo "  CP2K completed"
    else
      touch "$CALC_DIR/.failed"
      FAILED=1
      echo "  CP2K exited without normal end marker" >&2
    fi
  else
    touch "$CALC_DIR/.failed"
    FAILED=1
    echo "  CP2K/srun failed" >&2
  fi
  rm -f "$CALC_DIR/.running"
done

exit "$FAILED"
