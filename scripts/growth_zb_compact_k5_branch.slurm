#!/bin/bash
#SBATCH -J qd_compact_k5
#SBATCH -t 48:00:00
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem-per-cpu=2g
#SBATCH --qos=regular
#SBATCH --open-mode=append

# EXPERIMENT: does compactness-ranked parent selection below k=5 keep the
# lineage that leads to the k=13 Wulff core?
#
# Runs k=1 -> 6 under growth_compact_k5.yaml, which differs from the frozen
# growth.yaml in exactly one key: parents.compact_rank_below_k = 5.
#
# WHY IT STARTS AT k=1 AND NOT k=4.  The change alters which k=4 occupations
# are selected as parents, and the carrier of interest (k004_p006_Zf067f7ee)
# is absent from the previous run's index.csv -- it relaxed off-path and was
# never indexed, so it cannot be loaded as a parent from that tree.  k=4 has
# to be regenerated.  Steps k=1..4 are the cheap ones (~5 h combined in the
# k1_to_k8 run), so this is affordable.
#
# WHY IT STOPS AT k=6.  The lineage died at the k=4 -> 5 step.  Reaching k=6
# tests that step and the one after it, which is the whole question; k=6 -> 7
# alone cost 27 h in the previous run and tells you nothing new if the
# lineage is already gone.  Extend --k-to only if the readout says it lived.
#
# SEPARATE OUTPUT.  Never point this at growth_lattice_motifs_k1_to_k8:
# growth.yaml is frozen so that run is comparable end to end, and mixing the
# two selection rules into one tree destroys that.
#
# READOUT, after (or during) the run:
#   PYTHONPATH=src python scripts/check_wulff_ancestry.py $OUT
# Expect "lineage lost at k=5" to become a larger k, or "still alive".
# Compare against the same command on growth_lattice_motifs_k1_to_k8.

export PYTHONUNBUFFERED=1
source /scicomp/builds/Rocky/8.7/Common/software/Mamba/23.11.0-0/etc/profile.d/conda.sh
mamba activate qd-builder

export PATH="/scratch/iinfante/xtb/xtb-6.7.1/bin:$PATH"
export XTBPATH="/scratch/iinfante/xtb/xtb-6.7.1/share/xtb"

# Do NOT export OMP_NUM_THREADS: xtb_relax sets it per child from
# QD_XTB_THREADS, and an exported value would multiply with the worker count.
export QD_XTB_WORKERS=12
export QD_XTB_THREADS=2
export QD_EMBED_WORKERS=24        # also the fallback for QD_GRAPH_WORKERS

export OMP_STACKSIZE=1G
ulimit -s unlimited

export TMPDIR=${SLURM_TMPDIR:-/scratch/iinfante/tmp}
mkdir -p "$TMPDIR"

ROOT=/scratch/iinfante/escience/QD_Builder
PACK=/scratch/iinfante/CP2K/Nucleation/xtb/CdSe/growth_zb
OUT=$PACK/growth_compact_k5_k1_to_k6
LOG=$PACK/logs/$(basename "$OUT").log

mkdir -p "$PACK/logs"
cd "$ROOT"

# The branch config lives in the repo; copy it next to the pack yamls the
# tool resolves against.
cp "$ROOT/geometry_packs/cdse_cdcl2_zb/growth_compact_k5.yaml" "$PACK/"

python -u "$ROOT/tools/run_molecular_growth.py" \
  --pack-dir "$PACK" \
  --growth   "$PACK/growth_compact_k5.yaml" \
  --parents  /scratch/iinfante/CP2K/Nucleation/xtb/CdSe/new_generator/runs/gxtb_cdse_target_k1k2_p1p5 \
  --k-from 1 --k-to 6 --p-parents all \
  --output   "$OUT" \
  >> "$LOG" 2>&1
RC=$?

echo "=== ancestry readout ===" >> "$LOG"
PYTHONPATH="$ROOT/src" python "$ROOT/scripts/check_wulff_ancestry.py" "$OUT" >> "$LOG" 2>&1

exit $RC
