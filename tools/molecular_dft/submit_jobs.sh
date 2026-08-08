#!/bin/bash
# Submit one independent SLURM job per structure in the manifest.
#
# Run from the DFT tree root (directory with manifest.tsv and k001/, k002/, ...).
#
# Usage:
#   ./submit_jobs.sh [manifest.tsv] [cp2k_one.slurm]
#
# Manifest order is preserved (k, then p, then structure_id), so jobs are
# submitted from k001/p001 upward.  Each line becomes its own `sbatch` so the
# scheduler can pack every free core with separate jobs.
#
# Optional environment:
#   SKIP_DONE=1   skip dirs that already have a finished cp2k_job.out (default 1)
#   DRY_RUN=1     print sbatch lines without submitting

set -euo pipefail

CURRENT_DIR=$(pwd)
MANIFEST=${1:-$CURRENT_DIR/manifest.tsv}
SLURM_SCRIPT=${2:-$CURRENT_DIR/cp2k_one.slurm}
SKIP_DONE=${SKIP_DONE:-1}
DRY_RUN=${DRY_RUN:-0}

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST" >&2
  exit 2
fi
if [[ ! -f "$SLURM_SCRIPT" ]]; then
  # Fall back to the copy next to this script (repo tools/molecular_dft/)
  HERE=$(cd "$(dirname "$0")" && pwd)
  if [[ -f "$HERE/cp2k_one.slurm" ]]; then
    SLURM_SCRIPT=$HERE/cp2k_one.slurm
  else
    echo "SLURM script not found: $SLURM_SCRIPT" >&2
    echo "Copy cp2k_one.slurm into this directory or pass its path as arg 2." >&2
    exit 2
  fi
fi

MANIFEST=$(realpath "$MANIFEST")
SLURM_SCRIPT=$(realpath "$SLURM_SCRIPT")
RUN_ROOT=$(dirname "$MANIFEST")

TOTAL=$(( $(wc -l < "$MANIFEST") - 1 ))
if (( TOTAL < 1 )); then
  echo "Manifest contains no structures: $MANIFEST" >&2
  exit 2
fi

echo "Root:       $RUN_ROOT"
echo "Manifest:   $MANIFEST"
echo "SLURM:      $SLURM_SCRIPT"
echo "Structures: $TOTAL"
echo "Mode:       one independent sbatch per structure (manifest order)"
echo

SUBMITTED=0
SKIPPED=0
FAILED=0
INDEX=0

# Read TSV after header; preserve order (k, p, id as written by the preparer).
while IFS=$'\t' read -r IDX K P STRUCTURE_ID BOX RUN_DIR SOURCE_XYZ; do
  # Skip header if present
  if [[ "$IDX" == "index" ]]; then
    continue
  fi
  INDEX=$((INDEX + 1))
  CALC_DIR="$RUN_ROOT/$RUN_DIR"

  if [[ ! -d "$CALC_DIR" ]]; then
    echo "[$INDEX/$TOTAL] MISSING dir: $CALC_DIR" >&2
    FAILED=$((FAILED + 1))
    continue
  fi
  if [[ ! -f "$CALC_DIR/cp2k_job.in" || ! -f "$CALC_DIR/start.xyz" ]]; then
    echo "[$INDEX/$TOTAL] MISSING inputs in $CALC_DIR" >&2
    FAILED=$((FAILED + 1))
    continue
  fi
  if [[ "$SKIP_DONE" == "1" ]] && [[ -f "$CALC_DIR/cp2k_job.out" ]] \
      && grep -q "PROGRAM ENDED" "$CALC_DIR/cp2k_job.out" 2>/dev/null; then
    echo "[$INDEX/$TOTAL] skip done  k=$K p=$P $STRUCTURE_ID"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  JOB_NAME="CdSe_k${K}_p${P}_${STRUCTURE_ID}"
  # Shorten job name for SLURM limits
  if (( ${#JOB_NAME} > 64 )); then
    JOB_NAME="CdSe_${STRUCTURE_ID}"
    JOB_NAME=${JOB_NAME:0:64}
  fi

  echo "[$INDEX/$TOTAL] submit  k=$K p=$P $STRUCTURE_ID  -> $CALC_DIR"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  DRY: sbatch -J $JOB_NAME --chdir=$CALC_DIR $SLURM_SCRIPT"
    SUBMITTED=$((SUBMITTED + 1))
    continue
  fi

  if sbatch \
      -J "$JOB_NAME" \
      --chdir="$CALC_DIR" \
      --export="ALL,CALC_DIR=$CALC_DIR" \
      "$SLURM_SCRIPT" > /tmp/cdse_sbatch_$$.out 2>/tmp/cdse_sbatch_$$.err; then
    cat /tmp/cdse_sbatch_$$.out
    SUBMITTED=$((SUBMITTED + 1))
  else
    echo "  sbatch failed:" >&2
    cat /tmp/cdse_sbatch_$$.err >&2
    FAILED=$((FAILED + 1))
  fi
  rm -f /tmp/cdse_sbatch_$$.out /tmp/cdse_sbatch_$$.err
done < "$MANIFEST"

echo
echo "Done. submitted=$SUBMITTED skipped=$SKIPPED failed=$FAILED (of $TOTAL)"
if (( FAILED > 0 )); then
  exit 1
fi
exit 0
