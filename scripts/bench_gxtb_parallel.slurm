#!/bin/bash
#SBATCH -J bench_gxtb_parallel
#SBATCH -t 04:00:00
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem-per-cpu=2g
#SBATCH --qos=regular

# Find the best (concurrent g-xTB jobs) x (threads per job) on this node, at
# several cluster sizes.  One task, many cpus, one python process spawning N
# single-threaded children -- NOT N slurm tasks, because the growth driver
# holds index.csv, the manifests and parent selection in a single process.
#
# NOTE: deliberately does *not* export OMP_NUM_THREADS.  The benchmark sets it
# per grid point, and a global value would silently pin every grid point to the
# same thread count.  This is the same trap the production script falls into:
# xtb_relax._cli_env uses env.setdefault("OMP_NUM_THREADS", "1"), so an
# exported value is inherited rather than overridden -- with OMP_NUM_THREADS=16
# in the environment every g-xTB call runs 16-threaded, one at a time.

export PYTHONUNBUFFERED=1
source /scicomp/builds/Rocky/8.7/Common/software/Mamba/23.11.0-0/etc/profile.d/conda.sh
mamba activate qd-builder

export PATH="/scratch/iinfante/xtb/xtb-6.7.1/bin:$PATH"
export XTBPATH="/scratch/iinfante/xtb/xtb-6.7.1/share/xtb"

# 24 scratch dirs at once; keep them off the node's /tmp.
export TMPDIR=/scratch/iinfante/tmp
mkdir -p "$TMPDIR"

ROOT=/scratch/iinfante/escience/QD_Builder
PACK=/scratch/iinfante/CP2K/Nucleation/xtb/CdSe/growth_zb
RUN=$PACK/growth_k1_to_k5_zb4          # any finished run with k2..k5 bins
OUT=$PACK/bench_gxtb

mkdir -p "$OUT" "$PACK/logs"
cd "$ROOT"

python -u "$ROOT/tools/bench_gxtb_parallel.py" \
  --pack-dir "$PACK" \
  --run      "$RUN" \
  --bins 2,2 3,3 4,3 5,3 4,6 \
  --cores 24 \
  --concurrency 1,2,4,8,12,16,20,24 \
  --threads 1,2,4 \
  --repeats 2 \
  --amplitude 0.25 \
  --tmpdir "$TMPDIR" \
  --output   "$OUT" \
  > "$PACK/logs/bench_gxtb_parallel.log" 2>&1
