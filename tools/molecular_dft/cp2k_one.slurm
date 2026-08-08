#!/bin/bash
#SBATCH -J CdSe_mol
#SBATCH -t 24:00:00
#SBATCH -n 24
#SBATCH --mem-per-cpu=2g
#SBATCH --qos=regular

# One structure, one job.  Run from the isomer directory (or set CALC_DIR).
# Expects: cp2k_job.in and start.xyz in the working directory.

set -uo pipefail
module purge
module load CP2K/2024.3-foss-2023a
export OMP_NUM_THREADS=1

CALC_DIR=${CALC_DIR:-$(pwd)}
cd "$CALC_DIR" || exit 2

if [[ ! -f cp2k_job.in || ! -f start.xyz ]]; then
  echo "Missing cp2k_job.in or start.xyz in $CALC_DIR" >&2
  exit 2
fi

if [[ -f cp2k_job.out ]] && grep -q "PROGRAM ENDED" cp2k_job.out; then
  touch .done
  echo "Already completed: $CALC_DIR"
  exit 0
fi

rm -f .done .failed .running
touch .running
if srun cp2k.popt -i cp2k_job.in -o cp2k_job.out; then
  if grep -q "PROGRAM ENDED" cp2k_job.out; then
    touch .done
    echo "CP2K completed: $CALC_DIR"
    rm -f .running
    exit 0
  fi
  touch .failed
  rm -f .running
  echo "CP2K exited without PROGRAM ENDED: $CALC_DIR" >&2
  exit 1
fi
touch .failed
rm -f .running
echo "CP2K/srun failed: $CALC_DIR" >&2
exit 1
