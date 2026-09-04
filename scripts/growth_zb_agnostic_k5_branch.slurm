#!/bin/bash
#SBATCH -J qd_agnostic_k5
#SBATCH -t 48:00:00
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem-per-cpu=2g
#SBATCH --qos=regular
#SBATCH --open-mode=append

# EXPERIMENT: Target-agnostic lineage preservation.
# 1. Compactness ranking below k=5 deduplicating on canonical anion keys.
# 2. Widened low-k cap (cap=12 for k=1..4).
# 3. Reserved compact seats (3 seats for k=2..4, 2 seats for k>=5).
# 4. Cross-p retention of backbones stabilized by high-p passivation into lean bins.
#
# Runs k=1 -> 6 under growth_agnostic_k5.yaml.
#
# SEPARATE OUTPUT: Writes to $PACK/growth_agnostic_k5_k1_to_k6.
#
# READOUT:
#   PYTHONPATH=src python scripts/check_wulff_ancestry.py $OUT

export PYTHONUNBUFFERED=1
source /scicomp/builds/Rocky/8.7/Common/software/Mamba/23.11.0-0/etc/profile.d/conda.sh
mamba activate qd-builder

export PATH="/scratch/iinfante/xtb/xtb-6.7.1/bin:$PATH"
export XTBPATH="/scratch/iinfante/xtb/xtb-6.7.1/share/xtb"

export QD_XTB_WORKERS=12
export QD_XTB_THREADS=2
export QD_EMBED_WORKERS=24

export OMP_STACKSIZE=1G
ulimit -s unlimited

export TMPDIR=${SLURM_TMPDIR:-/scratch/iinfante/tmp}
mkdir -p "$TMPDIR"

ROOT=/scratch/iinfante/escience/QD_Builder
PACK=/scratch/iinfante/CP2K/Nucleation/xtb/CdSe/growth_zb
OUT=$PACK/growth_agnostic_k5_k1_to_k6
LOG=$PACK/logs/$(basename "$OUT").log

mkdir -p "$PACK/logs"
cd "$ROOT"

cp "$ROOT/geometry_packs/cdse_cdcl2_zb/growth_agnostic_k5.yaml" "$PACK/"

python -u "$ROOT/tools/run_molecular_growth.py" \
  --pack-dir "$PACK" \
  --growth   "$PACK/growth_agnostic_k5.yaml" \
  --parents  /scratch/iinfante/CP2K/Nucleation/xtb/CdSe/new_generator/runs/gxtb_cdse_target_k1k2_p1p5 \
  --k-from 1 --k-to 6 --p-parents all \
  --output   "$OUT" \
  >> "$LOG" 2>&1
RC=$?

echo "=== ancestry readout ===" >> "$LOG"
PYTHONPATH="$ROOT/src" python "$ROOT/scripts/check_wulff_ancestry.py" "$OUT" >> "$LOG" 2>&1

exit $RC
