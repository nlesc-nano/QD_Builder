#!/bin/bash
#SBATCH -J InAs_DFT
#SBATCH -t 24:00:00
#SBATCH -n 24
#SBATCH --mem-per-cpu=2g
#SBATCH --qos=regular

set -euo pipefail

module purge
module load CP2K/2024.3-foss-2023a
export OMP_NUM_THREADS=1

if (( $# != 2 )); then
  echo "Usage: sbatch cp2k.slurm START_FRAME END_FRAME" >&2
  exit 2
fi

START_FRAME=$1
END_FRAME=$2
if ! [[ "$START_FRAME" =~ ^[1-9][0-9]*$ && "$END_FRAME" =~ ^[1-9][0-9]*$ ]]; then
  echo "Frame bounds must be positive integers" >&2
  exit 2
fi
if (( START_FRAME > END_FRAME )); then
  echo "START_FRAME must not exceed END_FRAME" >&2
  exit 2
fi

ROOT=$(pwd)
META="$ROOT/frames.tsv"
TEMPLATE="$ROOT/template_inas_opt.in"
BATCH_DIR="$ROOT/batch_${START_FRAME}_to_${END_FRAME}"
BATCH_WFN="$BATCH_DIR/InAs-RESTART_batch.wfn"

[[ -f "$META" ]] || { echo "Missing $META" >&2; exit 2; }
[[ -f "$TEMPLATE" ]] || { echo "Missing $TEMPLATE" >&2; exit 2; }
mkdir -p "$BATCH_DIR"

# Set USE_ROLLING_WFN=0 for independent isomer calculations (recommended for
# energy comparisons). Set it to 1 only if a rolling restart is desired.
USE_ROLLING_WFN=${USE_ROLLING_WFN:-0}
if [[ "$USE_ROLLING_WFN" == "1" ]]; then
  if [[ -f "$BATCH_WFN" ]]; then
    echo "Resuming batch from $BATCH_WFN"
  elif [[ -f "$ROOT/InAs-RESTART.wfn" ]]; then
    cp "$ROOT/InAs-RESTART.wfn" "$BATCH_WFN"
    echo "Seeded batch WFN from $ROOT/InAs-RESTART.wfn"
  else
    echo "No WFN seed found; first frame uses an atomic guess"
  fi
else
  echo "Independent isomer mode: atomic SCF guess for every frame"
fi

lookup_row() {
  awk -F '\t' -v frame="$1" 'NR > 1 && $1 == frame { print; exit }' "$META"
}

for ((i=START_FRAME; i<=END_FRAME; i++)); do
  ROW=$(lookup_row "$i")
  if [[ -z "$ROW" ]]; then
    echo "No metadata row for frame $i" >&2
    exit 1
  fi
  IFS=$'\t' read -r FRAME K P STRUCTURE_ID BOX XYZ_FILE SOURCE_XYZ <<< "$ROW"
  PREFIX="InAs_frame_${FRAME}"
  RUN_DIR="$BATCH_DIR/$PREFIX"
  mkdir -p "$RUN_DIR"

  if [[ -f "$RUN_DIR/${PREFIX}.out" ]] && grep -q "PROGRAM ENDED" "$RUN_DIR/${PREFIX}.out"; then
    echo "Frame $FRAME (k=$K p=$P) already completed; skipping"
    continue
  fi

  if [[ ! -f "$ROOT/$XYZ_FILE" ]]; then
    echo "Missing $ROOT/$XYZ_FILE" >&2
    exit 1
  fi
  cp "$ROOT/$XYZ_FILE" "$RUN_DIR/start.xyz"

  if [[ "$USE_ROLLING_WFN" == "1" && -f "$BATCH_WFN" ]]; then
    SCF_GUESS=restart
    WFN_LINE="    WFN_RESTART_FILE_NAME $BATCH_WFN"
  else
    SCF_GUESS=atomic
    WFN_LINE=""
  fi

  sed \
    -e "s|__XYZ_FILE__|start.xyz|g" \
    -e "s|__PREFIX__|$PREFIX|g" \
    -e "s|__ABC_SIZE__|$BOX $BOX $BOX|g" \
    -e "s|__SCF_GUESS__|$SCF_GUESS|g" \
    -e "s|__WFN_FILE__|$WFN_LINE|g" \
    "$TEMPLATE" > "$RUN_DIR/${PREFIX}.in"

  echo "Starting frame $FRAME: k=$K p=$P id=$STRUCTURE_ID box=${BOX} A"
  echo "Source: $SOURCE_XYZ"
  (
    cd "$RUN_DIR"
    srun cp2k.popt "${PREFIX}.in" > "${PREFIX}.out"
  )

  if grep -q "PROGRAM ENDED" "$RUN_DIR/${PREFIX}.out"; then
    touch "$RUN_DIR/.done"
    if [[ "$USE_ROLLING_WFN" == "1" && -f "$RUN_DIR/${PREFIX}-RESTART.wfn" ]]; then
      cp "$RUN_DIR/${PREFIX}-RESTART.wfn" "$BATCH_WFN"
    fi
    echo "Frame $FRAME completed"
  else
    touch "$RUN_DIR/.failed"
    echo "Frame $FRAME failed or hit walltime; stopping batch" >&2
    exit 1
  fi
done

if [[ "$USE_ROLLING_WFN" == "1" ]]; then
  rm -f "$BATCH_WFN"
fi
echo "Batch ${START_FRAME}-${END_FRAME} completed"
