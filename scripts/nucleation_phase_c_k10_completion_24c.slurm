#!/bin/bash
#SBATCH -J qd_nuc_k10cmp
#SBATCH -t 48:00:00
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem-per-cpu=2g
#SBATCH --qos=regular
#SBATCH --open-mode=append
#SBATCH -o nucleation_k10cmp_%j.out
#SBATCH -e nucleation_k10cmp_%j.err

# Automated k=10 completion:
#   1. write high-p geometry diagnostics;
#   2. create a p11--p13 overlap-relaxed policy view without new xTB calls;
#   3. run strict fixed-k p1--p10 exploration;
#   4. run overlap-relaxed fixed-k p11--p13 exploration;
#   5. merge both branches into the canonical k=10 archive.

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
BASE=${QD_GROWTH_ZB:-/scratch/iinfante/CP2K/Nucleation/xtb/CdSe/growth_zb}
SOURCE=${QD_SOURCE:-$BASE/nucleation_phase_c_k10_all_p_v1}
VALIDATION=${QD_VALIDATION_OUT:-$BASE/nucleation_phase_c_k10_transition_validation_v1}
POLICY=${QD_POLICY_OUT:-$BASE/nucleation_phase_c_k10_policy_view_v1}
STRICT=${QD_STRICT_OUT:-$BASE/nucleation_phase_c_k10_fixed_strict_v1}
RELAXED=${QD_RELAXED_OUT:-$BASE/nucleation_phase_c_k10_fixed_relaxed_high_p_v1}
CANONICAL=${QD_OUT:-$BASE/nucleation_phase_c_k10_canonical_v1}
PACK="$ROOT/geometry_packs/cdse_cdcl2_zb"
GROWTH="$PACK/growth_agnostic_k5.yaml"
STRICT_CONFIG="$PACK/adaptive_phase_c_k10_fixed_strict.yaml"
RELAXED_CONFIG="$PACK/adaptive_phase_c_k10_fixed_relaxed_high_p.yaml"

export TMPDIR=${SLURM_TMPDIR:-/scratch/iinfante/tmp/nucleation_k10cmp_${SLURM_JOB_ID}}
mkdir -p "$TMPDIR" "$VALIDATION" "$(dirname "$CANONICAL")"

echo "[workflow] source=$SOURCE"
echo "[workflow] policy=$POLICY"
echo "[workflow] strict=$STRICT"
echo "[workflow] relaxed=$RELAXED"
echo "[workflow] canonical=$CANONICAL"
if [[ ${SLURM_CPUS_PER_TASK:-0} -lt 24 ]]; then
    echo "[workflow] ERROR: configuration requires 24 CPUs per task"
    exit 2
fi
if [[ ! -f "$SOURCE/minima.json" || ! -f "$SOURCE/events.jsonl" ]]; then
    echo "[workflow] ERROR: source minima/journal not found: $SOURCE"
    exit 2
fi
for executable in gxtb flock; do
    if ! command -v "$executable" >/dev/null 2>&1; then
        echo "[workflow] ERROR: $executable is not available"
        exit 2
    fi
done

exec 9>"${CANONICAL}.workflow.lock"
if ! flock -n 9; then
    echo "[workflow] ERROR: another process holds ${CANONICAL}.workflow.lock"
    exit 3
fi
cd "$ROOT"

echo "[workflow] step 1/5: k10 p11-p13 transition diagnostics"
python -u "$ROOT/tools/validate_nucleation_transition.py" "$SOURCE" \
    --output "$VALIDATION" --k 10 --p 11 12 13 --top 24

echo "[workflow] step 2/5: overlap-only high-p policy view"
if [[ ! -f "$POLICY/minima.json" ]]; then
    python -u "$ROOT/tools/build_adaptive_policy_view.py" \
        --source "$SOURCE" --output "$POLICY" --k 10 --p 11 12 13 \
        --allow-violation mu3_host_bridge_overlap \
        --classification-mode relaxed_mu3_overlap_k10_high_p
else
    echo "[workflow] policy view already present; reusing $POLICY"
fi

run_adaptive () {
    local label=$1
    local config=$2
    local output=$3
    echo "[workflow] validating $label configuration"
    PYTHONPATH="$ROOT/src" python -u "$ROOT/tools/run_adaptive_nucleation.py" \
        --config "$config" --pack-dir "$PACK" --growth "$GROWTH" \
        --source "$POLICY" --output "$output" --dry-run
    echo "[workflow] running $label"
    PYTHONPATH="$ROOT/src" python -u "$ROOT/tools/run_adaptive_nucleation.py" \
        --config "$config" --pack-dir "$PACK" --growth "$GROWTH" \
        --source "$POLICY" --output "$output"
}

echo "[workflow] step 3/5: strict fixed-k10 p1-p10"
run_adaptive strict-p1-p10 "$STRICT_CONFIG" "$STRICT"

echo "[workflow] step 4/5: overlap-relaxed fixed-k10 p11-p13"
run_adaptive relaxed-p11-p13 "$RELAXED_CONFIG" "$RELAXED"

echo "[workflow] step 5/5: canonical branch merge"
if [[ ! -f "$CANONICAL/minima.json" ]]; then
    PYTHONPATH="$ROOT/src" python -u "$ROOT/tools/merge_adaptive_archives.py" \
        --source "$STRICT" --source "$RELAXED" --output "$CANONICAL" \
        --pack "$PACK/run_gxtb.yaml"
else
    echo "[workflow] canonical archive already present; leaving it unchanged"
fi

echo "[workflow] completed at $(date -Is)"
