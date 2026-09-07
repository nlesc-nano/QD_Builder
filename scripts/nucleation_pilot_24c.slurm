#!/bin/bash
#SBATCH -J qd_nuc_pilot
#SBATCH -t 48:00:00
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem-per-cpu=2g
#SBATCH --qos=regular
#SBATCH --open-mode=append

# Bounded k<=6 control/experimental nucleation pilot. The internal launch
# cutoff is 46 h, leaving two hours inside this allocation for the final
# <=30 min g-xTB batch, checkpointing and report generation.
#
# Submit:
#   sbatch scripts/nucleation_pilot_24c.slurm
#
# Override paths when needed:
#   sbatch --export=ALL,QD_ROOT=/path/to/QD_Builder,QD_SEEDS=/path/to/seeds,QD_OUT=/path/to/output \
#     scripts/nucleation_pilot_24c.slurm
#
# Resubmit the identical command with the identical QD_OUT to resume. Never
# run two jobs against one output directory: the lock below rejects this.

set -euo pipefail
export PYTHONUNBUFFERED=1

source /scicomp/builds/Rocky/8.7/Common/software/Mamba/23.11.0-0/etc/profile.d/conda.sh
mamba activate qd-builder

export PATH="/scratch/iinfante/xtb/xtb-6.7.1/bin:$PATH"
export XTBPATH="/scratch/iinfante/xtb/xtb-6.7.1/share/xtb"

# PilotConfig runs 24 concurrent backend calls, one thread per call.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_STACKSIZE=1G
ulimit -s unlimited

ROOT=${QD_ROOT:-/scratch/iinfante/escience/QD_Builder}
SEEDS=${QD_SEEDS:-/scratch/iinfante/CP2K/Nucleation/xtb/CdSe/new_generator/runs/gxtb_cdse_target_k1k2_p1p5}
OUT=${QD_OUT:-/scratch/iinfante/CP2K/Nucleation/xtb/CdSe/growth_zb/nucleation_pilot_v1}
CONFIG="$ROOT/geometry_packs/cdse_cdcl2_zb/pilot.yaml"
PACK="$ROOT/geometry_packs/cdse_cdcl2_zb"
GROWTH="$PACK/growth_agnostic_k5.yaml"

export TMPDIR=${SLURM_TMPDIR:-/scratch/iinfante/tmp/nucleation_pilot_${SLURM_JOB_ID}}
mkdir -p "$TMPDIR" "$OUT" "$OUT/logs"
LOG="$OUT/logs/slurm_${SLURM_JOB_ID}.log"
exec >> "$LOG" 2>&1

echo "[pilot] start $(date -Is), job ${SLURM_JOB_ID}, host $(hostname)"
echo "[pilot] root=$ROOT"
echo "[pilot] seeds=$SEEDS"
echo "[pilot] output=$OUT"

if [[ ${SLURM_CPUS_PER_TASK:-0} -lt 24 ]]; then
    echo "[pilot] ERROR: this configuration requires 24 CPUs per task"
    exit 2
fi
if [[ ! -f "$SEEDS/index.csv" ]]; then
    echo "[pilot] ERROR: seed index not found: $SEEDS/index.csv"
    exit 2
fi
if ! command -v gxtb >/dev/null 2>&1; then
    echo "[pilot] ERROR: gxtb is not available on PATH"
    exit 2
fi
if ! command -v flock >/dev/null 2>&1; then
    echo "[pilot] ERROR: flock is required to prevent concurrent writers"
    exit 2
fi

exec 9>"$OUT/.pilot.lock"
if ! flock -n 9; then
    echo "[pilot] ERROR: another pilot process holds $OUT/.pilot.lock"
    exit 3
fi

cd "$ROOT"

# Configuration/path validation only; this performs no calculation or write.
PYTHONPATH="$ROOT/src" python -u "$ROOT/tools/run_nucleation_pilot.py" \
    --config "$CONFIG" \
    --pack-dir "$PACK" \
    --growth "$GROWTH" \
    --seeds "$SEEDS" \
    --output "$OUT" \
    --dry-run

set +e
PYTHONPATH="$ROOT/src" python -u "$ROOT/tools/run_nucleation_pilot.py" \
    --config "$CONFIG" \
    --pack-dir "$PACK" \
    --growth "$GROWTH" \
    --seeds "$SEEDS" \
    --output "$OUT"
RC=$?
set -e
echo "[pilot] exit $RC at $(date -Is)"
exit "$RC"
