#!/bin/bash
#SBATCH -J qd_nuc_k10
#SBATCH -t 48:00:00
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem-per-cpu=2g
#SBATCH --qos=regular
#SBATCH --open-mode=append
#SBATCH -o nucleation_k10_%j.out
#SBATCH -e nucleation_k10_%j.err

# Targeted k=9 -> k=10 strict growth test.
# Override paths with:
# sbatch --export=ALL,QD_ROOT=...,QD_SOURCE=...,QD_OUT=... \
#   scripts/nucleation_phase_c_k10_24c.slurm

set -euo pipefail
export PYTHONUNBUFFERED=1
echo "[launcher] entered job ${SLURM_JOB_ID} at $(date -Is) on $(hostname)"

source /scicomp/builds/Rocky/8.7/Common/software/Mamba/23.11.0-0/etc/profile.d/conda.sh
conda activate qd-builder
echo "[launcher] activated qd-builder with python $(command -v python)"

export PATH="/scratch/iinfante/xtb/xtb-6.7.1/bin:$PATH"
export XTBPATH="/scratch/iinfante/xtb/xtb-6.7.1/share/xtb"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_STACKSIZE=1G
ulimit -s unlimited

ROOT=${QD_ROOT:-/scratch/iinfante/escience/QD_Builder}
SOURCE=${QD_SOURCE:-/scratch/iinfante/CP2K/Nucleation/xtb/CdSe/growth_zb/nucleation_phase_c_k9_canonical_v1}
OUT=${QD_OUT:-/scratch/iinfante/CP2K/Nucleation/xtb/CdSe/growth_zb/nucleation_phase_c_k10_v1}
CONFIG=${QD_CONFIG:-$ROOT/geometry_packs/cdse_cdcl2_zb/adaptive_phase_c_k10.yaml}
PACK="$ROOT/geometry_packs/cdse_cdcl2_zb"
GROWTH="$PACK/growth_agnostic_k5.yaml"

export TMPDIR=${SLURM_TMPDIR:-/scratch/iinfante/tmp/nucleation_k10_${SLURM_JOB_ID}}
mkdir -p "$TMPDIR" "$OUT"

echo "[adaptive] source=$SOURCE"
echo "[adaptive] output=$OUT"
echo "[adaptive] config=$CONFIG"
if [[ ${SLURM_CPUS_PER_TASK:-0} -lt 24 ]]; then
    echo "[adaptive] ERROR: configuration requires 24 CPUs per task"
    exit 2
fi
if [[ ! -f "$SOURCE/minima.json" || ! -f "$SOURCE/events.jsonl" ]]; then
    echo "[adaptive] ERROR: source minima/journal not found under: $SOURCE"
    exit 2
fi
if ! command -v gxtb >/dev/null 2>&1; then
    echo "[adaptive] ERROR: gxtb is not available on PATH"
    exit 2
fi
if ! command -v flock >/dev/null 2>&1; then
    echo "[adaptive] ERROR: flock is required"
    exit 2
fi

exec 9>"$OUT/.adaptive.lock"
if ! flock -n 9; then
    echo "[adaptive] ERROR: another process holds $OUT/.adaptive.lock"
    exit 3
fi

cd "$ROOT"
PYTHONPATH="$ROOT/src" python -u "$ROOT/tools/run_adaptive_nucleation.py" \
    --config "$CONFIG" --pack-dir "$PACK" --growth "$GROWTH" \
    --source "$SOURCE" --output "$OUT" --dry-run

set +e
PYTHONPATH="$ROOT/src" python -u "$ROOT/tools/run_adaptive_nucleation.py" \
    --config "$CONFIG" --pack-dir "$PACK" --growth "$GROWTH" \
    --source "$SOURCE" --output "$OUT"
RC=$?
set -e
echo "[adaptive] exit $RC at $(date -Is)"
exit "$RC"
