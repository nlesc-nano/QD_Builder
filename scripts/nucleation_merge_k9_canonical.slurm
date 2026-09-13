#!/bin/bash
#SBATCH -J qd_nuc_k9merge
#SBATCH -t 00:30:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8g
#SBATCH --qos=regular
#SBATCH --open-mode=append
#SBATCH -o nucleation_k9merge_%j.out
#SBATCH -e nucleation_k9merge_%j.err

# Consolidate the independent low-p growth and fixed-k=9 exploration branches.
# This job performs no g-xTB calls.

set -euo pipefail
export PYTHONUNBUFFERED=1
echo "[launcher] entered merge job ${SLURM_JOB_ID} at $(date -Is) on $(hostname)"

source /scicomp/builds/Rocky/8.7/Common/software/Mamba/23.11.0-0/etc/profile.d/conda.sh
conda activate qd-builder
echo "[launcher] activated qd-builder with python $(command -v python)"

ROOT=${QD_ROOT:-/scratch/iinfante/escience/QD_Builder}
BASE=${QD_GROWTH_ZB:-/scratch/iinfante/CP2K/Nucleation/xtb/CdSe/growth_zb}
LOW=${QD_LOW_SOURCE:-$BASE/nucleation_phase_c_k9_low_p_lineage_completion_v1}
FIXED=${QD_FIXED_SOURCE:-$BASE/nucleation_phase_c_k9_fixed_all_p_v1}
OUT=${QD_OUT:-$BASE/nucleation_phase_c_k9_canonical_v1}
PACK="$ROOT/geometry_packs/cdse_cdcl2_zb/run_gxtb.yaml"

echo "[merge] low-p growth source=$LOW"
echo "[merge] fixed-k source=$FIXED"
echo "[merge] canonical output=$OUT"
for source in "$LOW" "$FIXED"; do
    if [[ ! -f "$source/minima.json" ]]; then
        echo "[merge] ERROR: missing $source/minima.json"
        exit 2
    fi
done
if [[ -e "$OUT/minima.json" ]]; then
    echo "[merge] ERROR: output already contains minima.json: $OUT"
    exit 2
fi

mkdir -p "$(dirname "$OUT")"
exec 9>"${OUT}.lock"
if ! flock -n 9; then
    echo "[merge] ERROR: another process holds ${OUT}.lock"
    exit 3
fi

cd "$ROOT"
PYTHONPATH="$ROOT/src" python -u "$ROOT/tools/merge_adaptive_archives.py" \
    --source "$LOW" --source "$FIXED" --output "$OUT" --pack "$PACK" \
    --dry-run
PYTHONPATH="$ROOT/src" python -u "$ROOT/tools/merge_adaptive_archives.py" \
    --source "$LOW" --source "$FIXED" --output "$OUT" --pack "$PACK"

echo "[merge] completed at $(date -Is)"
