#!/bin/bash
#SBATCH -J grow_zb_24c
#SBATCH -t 24:00:00
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem-per-cpu=2g
#SBATCH --qos=regular

# 24 cores: 12 g-xTB jobs x 2 threads.
#
# Measured on this node over 10-26 atom clusters (geometric mean over 7 sizes,
# normalised to one single-threaded job):
#     16 cores:  16x1 10.09x  >  8x2 8.48x  >  4x4 7.00x
#     24 cores:  12x2 11.41x  ~= 24x1 11.19x
# The two are a tie on average, but 24x1 collapses on some sizes -- at 15 atoms
# it peaks at 16 concurrent (4.70 opt/s) and falls to 3.01 at 24, while 12x2
# holds 4.90.  12x2 is the safer side of that saturation.

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

# Node-local scratch if there is any: QD_XTB_WORKERS dirs are created and
# deleted per batch, and a network filesystem is a plausible reason the
# benchmark stops scaling past ~16 concurrent jobs.
export TMPDIR=${SLURM_TMPDIR:-/scratch/iinfante/tmp}
mkdir -p "$TMPDIR"

ROOT=/scratch/iinfante/escience/QD_Builder
PACK=/scratch/iinfante/CP2K/Nucleation/xtb/CdSe/growth_zb
OUT=$PACK/growth_k1_to_k5_zb6

mkdir -p "$PACK/logs"
cd "$ROOT"

python -u "$ROOT/tools/run_molecular_growth.py" \
  --pack-dir "$PACK" \
  --growth   "$PACK/growth.yaml" \
  --parents  /scratch/iinfante/CP2K/Nucleation/xtb/CdSe/new_generator/runs/gxtb_cdse_target_k1k2_p1p5 \
  --k-from 1 --k-to 5 --p-parents all \
  --output   "$OUT" \
  > "$PACK/logs/$(basename $OUT).log" 2>&1
