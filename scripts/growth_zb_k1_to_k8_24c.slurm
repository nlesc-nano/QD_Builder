#!/bin/bash
#SBATCH -J qd_geom_k1k8
#SBATCH -t 72:00:00
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem-per-cpu=2g
#SBATCH --qos=regular

# k=1 -> 8 with lattice μ2 pairs, geometric Cl seeds, embed.yaml morph
# (core free, clash-aware, short LS), then frozen-core g-xTB and full opt.
#
# Copy the pack yamls from the repo into $PACK before submitting:
#   cp geometry_packs/cdse_cdcl2_zb/{growth,graph_rules,embed,run_gxtb}.yaml $PACK/
#
# Watch:
#   embed=geom freeze_core=1
#   grep -c "propagation_eligible=true"
#   max Cd-Se CN of eligible endpoints (was pinned at 3)

export PYTHONUNBUFFERED=1
source /scicomp/builds/Rocky/8.7/Common/software/Mamba/23.11.0-0/etc/profile.d/conda.sh
mamba activate qd-builder

export PATH="/scratch/iinfante/xtb/xtb-6.7.1/bin:$PATH"
export XTBPATH="/scratch/iinfante/xtb/xtb-6.7.1/share/xtb"

# Do NOT export OMP_NUM_THREADS here.  xtb_relax sets it per child from
# QD_XTB_THREADS; an exported value would be inherited by every g-xTB call and
# multiply with the worker count.
export QD_XTB_WORKERS=12          # concurrent g-xTB processes
export QD_XTB_THREADS=2           # OpenMP threads inside each one
# Embedding is single-threaded Python, so it gets one process per core.
export QD_EMBED_WORKERS=24

export OMP_STACKSIZE=1G
ulimit -s unlimited

export TMPDIR=${SLURM_TMPDIR:-/scratch/iinfante/tmp}
mkdir -p "$TMPDIR"

ROOT=/scratch/iinfante/escience/QD_Builder
PACK=/scratch/iinfante/CP2K/Nucleation/xtb/CdSe/growth_zb
OUT=$PACK/growth_geom_k1_to_k8

mkdir -p "$PACK/logs"
cd "$ROOT"

python -u "$ROOT/tools/run_molecular_growth.py" \
  --pack-dir "$PACK" \
  --growth   "$PACK/growth.yaml" \
  --parents  /scratch/iinfante/CP2K/Nucleation/xtb/CdSe/new_generator/runs/gxtb_cdse_target_k1k2_p1p5 \
  --k-from 1 --k-to 8 --p-parents all \
  --output   "$OUT" \
  > "$PACK/logs/$(basename $OUT).log" 2>&1
