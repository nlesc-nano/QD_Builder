#!/bin/bash
#SBATCH -J CdSe_DFT
#SBATCH -t 24:00:00
#SBATCH -n 24
##SBATCH -c 1
#SBATCH --mem-per-cpu=2g
#SBATCH --qos=regular
#SBATCH --array=0-0

set -uo pipefail

module purge
module load CP2K/2024.3-foss-2023a

export OMP_NUM_THREADS=1
NP=${SLURM_NTASKS:-24}
echo "$NP processes"

# MANIFEST is produced by prepare_cp2k_runs.py. BATCH_SIZE structures are run
# sequentially inside each array task. Command-line --array overrides the safe
# single-task default above.
MANIFEST=${MANIFEST:-runs/cdse_map/dft_all/manifest.tsv}
BATCH_SIZE=${BATCH_SIZE:-1}
BATCH_ID=${SLURM_ARRAY_TASK_ID:-0}

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST" >&2
  exit 2
fi
if ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH_SIZE must be a positive integer; got $BATCH_SIZE" >&2
  exit 2
fi

MANIFEST=$(realpath "$MANIFEST")
RUN_ROOT=$(dirname "$MANIFEST")
mapfile -t ROWS < <(tail -n +2 "$MANIFEST")
TOTAL=${#ROWS[@]}
FIRST=$((BATCH_ID * BATCH_SIZE))
LAST=$((FIRST + BATCH_SIZE - 1))

if (( FIRST >= TOTAL )); then
  echo "Batch $BATCH_ID starts at $FIRST but manifest has $TOTAL structures"
  exit 0
fi
if (( LAST >= TOTAL )); then
  LAST=$((TOTAL - 1))
fi

echo "Batch $BATCH_ID: manifest rows $FIRST through $LAST of $TOTAL"
BATCH_FAILED=0
for ((ROW_INDEX=FIRST; ROW_INDEX<=LAST; ROW_INDEX++)); do
  IFS=$'\t' read -r INDEX K P STRUCTURE_ID BOX RUN_DIR SOURCE_XYZ <<< "${ROWS[$ROW_INDEX]}"
  CALC_DIR="$RUN_ROOT/$RUN_DIR"
  echo "[$INDEX/$TOTAL] k=$K p=$P id=$STRUCTURE_ID box=$BOX A"
  echo "  source: $SOURCE_XYZ"
  echo "  run:    $CALC_DIR"

  if [[ ! -f "$CALC_DIR/cp2k_job.in" || ! -f "$CALC_DIR/start.xyz" ]]; then
    echo "Missing prepared input under $CALC_DIR" >&2
    BATCH_FAILED=1
    continue
  fi
  if [[ -f "$CALC_DIR/cp2k_job.out" ]] && grep -q "PROGRAM ENDED AT" "$CALC_DIR/cp2k_job.out"; then
    echo "  already completed; skipping"
    touch "$CALC_DIR/.done"
    continue
  fi

  cd "$CALC_DIR" || exit 2
  rm -f .done .failed .running .scf_not_converged
  touch .running
  if srun cp2k.popt -i cp2k_job.in -o cp2k_job.out; then
    if grep -q "PROGRAM ENDED AT" cp2k_job.out; then
      touch .done
      echo "  CP2K completed"
    else
      touch .failed
      BATCH_FAILED=1
      echo "  CP2K exited without its normal end marker" >&2
    fi
  else
    touch .failed
    BATCH_FAILED=1
    echo "  CP2K/srun failed" >&2
  fi
  if grep -q "SCF run NOT converged" cp2k_job.out 2>/dev/null; then
    touch .scf_not_converged
    echo "  WARNING: at least one SCF cycle did not converge" >&2
  fi
  rm -f .running
  cd "$SLURM_SUBMIT_DIR" || exit 2
done

exit "$BATCH_FAILED"
