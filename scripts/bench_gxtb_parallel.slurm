#!/bin/bash
#SBATCH -J bench_gxtb_parallel
#SBATCH -t 04:00:00
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem-per-cpu=2g
#SBATCH --qos=regular

# Find the best (concurrent g-xTB jobs) x (threads per job) on this node, at
# several cluster sizes.
#
# One task, many cpus, one python process spawning N single-threaded children
# -- NOT N slurm tasks, because the growth driver holds index.csv, the
# manifests and parent selection in a single process.
#
# IMPORTANT: this deliberately does *not* export OMP_NUM_THREADS.  The
# benchmark sets it per grid point, and a global value would pin every point to
# the same thread count, silently turning the --threads axis into a no-op.
#
# The same trap is live in the production growth job: xtb_relax._cli_env uses
#     env.setdefault("OMP_NUM_THREADS", "1")
# so an exported value is *inherited*, not overridden.  With
# OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK every g-xTB call runs 16-threaded, one
# at a time -- and threading a 17-atom optimisation is worth about 1.3x.

export PYTHONUNBUFFERED=1
source /scicomp/builds/Rocky/8.7/Common/software/Mamba/23.11.0-0/etc/profile.d/conda.sh
mamba activate qd-builder

export PATH="/scratch/iinfante/xtb/xtb-6.7.1/bin:$PATH"
export XTBPATH="/scratch/iinfante/xtb/xtb-6.7.1/share/xtb"

# Up to --cores scratch directories exist at once; keep them off node /tmp.
export TMPDIR=/scratch/iinfante/tmp
mkdir -p "$TMPDIR"

ROOT=/scratch/iinfante/escience/QD_Builder
PACK=/scratch/iinfante/CP2K/Nucleation/xtb/CdSe/growth_zb
OUT=$PACK/bench_gxtb

# Size spread is the point of this benchmark, so use the run with the widest
# one.  growth_k1_to_k4_zb2 predates the max_p_child cap and is the only run
# holding k4p4/p5/p6, giving 10 -> 26 atoms:
#   k2p2 10   k3p3 15   k4p3 17   k4p4 20   k4p5 23   k4p6 26
# zb3/zb4/zb5 stop at p=3 and would only span 10 -> 19.
RUN=$PACK/growth_k1_to_k4_zb2
BINS="2,2 3,3 4,3 4,4 4,5 4,6"

# A k=5 point from the newer run, if it is there (19 atoms, 5 Se in the core).
EXTRA=""
K5=$(ls "$PACK"/growth_k1_to_k5_zb4/k005/p003/*_xtb.xyz 2>/dev/null | grep -v ' 2\.' | head -1)
[ -n "$K5" ] && EXTRA="--structures $K5"

if [ ! -d "$RUN" ]; then
  echo "ERROR: reference run not found: $RUN" >&2
  echo "Point RUN at any finished growth run and set BINS to bins it has." >&2
  exit 1
fi

mkdir -p "$OUT" "$PACK/logs"
cd "$ROOT"

python -u "$ROOT/tools/bench_gxtb_parallel.py" \
  --pack-dir "$PACK" \
  --run      "$RUN" \
  --bins $BINS \
  $EXTRA \
  --cores 24 \
  --concurrency 1,2,4,8,12,16,20,24 \
  --threads 1,2,4 \
  --repeats 2 \
  --amplitude 0.25 \
  --tmpdir "$TMPDIR" \
  --output   "$OUT" \
  > "$PACK/logs/bench_gxtb_parallel.log" 2>&1
