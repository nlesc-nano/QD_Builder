#!/bin/bash
#SBATCH -J qd_motifs_k1k13
#SBATCH -t 72:00:00
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem-per-cpu=2g
#SBATCH --qos=regular
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@900

# Resume growth_lattice_motifs_k1_to_k8 after the kill mid k=7, and carry it
# on to k = 13.  Self-chaining: one 72 h job will not reach k=13, so on the
# walltime signal this stops the worker cleanly and submits the next link.
#
# WHAT RESUMES, AND AT WHAT GRANULARITY
#   * finished k-steps          .step_kXXX_complete            (k=1…6)
#   * finished move-A bins      .bin_A_complete
#   * finished move-B opts      index.csv / *_xtb.xyz
#   * finished move-Z bins      zb_motifs.jsonl                (k=7 p=1…9)
# The last one is new on branch `nucleation`; without it a restart re-opts
# every k=7 bin from p=1.  git pull on the HPC before submitting.
#
# --k-from is the *parent* k.  Keep it at 1.  --k-from 7 would start the
# k=7 -> 8 step and silently skip the unfinished k=7 p=10 bin.
#
# --output MUST stay the existing directory.  Rename it and every marker,
# index row and zb_motifs.jsonl line above is invisible: the run starts over.
#
# Copy the pack yamls from the repo into $PACK before submitting:
#   cp geometry_packs/cdse_cdcl2_zb/{growth,graph_rules,embed,run_gxtb}.yaml $PACK/
#   cp geometry_packs/cdse_cdcl2_zb/k13_wulff_core.yaml $PACK/
# growth.yaml is unchanged from the k=8 run -- its by_k block k_min 4 / k_max 12
# is keyed on the *parent* k, so it already covers the last step k=12 -> 13.
#
# Watch:
#   grep 'RESUME: skip move Z' $PACK/logs/growth_lattice_motifs_k1_to_k8.log
#   grep 'zb-motifs'          $PACK/logs/growth_lattice_motifs_k1_to_k8.log
#   grep 'endpoint'           $PACK/logs/growth_lattice_motifs_k1_to_k8.log
#   ls $OUT/.step_k*_complete

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
export QD_EMBED_WORKERS=24        # embedding is single-threaded Python

export OMP_STACKSIZE=1G
ulimit -s unlimited

export TMPDIR=${SLURM_TMPDIR:-/scratch/iinfante/tmp}
mkdir -p "$TMPDIR"

ROOT=/scratch/iinfante/escience/QD_Builder
PACK=/scratch/iinfante/CP2K/Nucleation/xtb/CdSe/growth_zb
OUT=$PACK/growth_lattice_motifs_k1_to_k8
LOG=$PACK/logs/$(basename "$OUT").log

SCRIPT=$ROOT/scripts/growth_zb_k1_to_k13_resume.slurm
LINK=${QD_CHAIN_LINK:-1}
MAX_LINKS=8
DONE_MARKER=$OUT/.step_k012_complete

mkdir -p "$PACK/logs"
cd "$ROOT"

echo "[chain] link $LINK/$MAX_LINKS  job $SLURM_JOB_ID  $(date -Is)" >> "$LOG"

# On the walltime signal: stop the worker so it cannot overlap with the next
# link writing into the same $OUT, then submit that link.
requeue() {
    echo "[chain] walltime signal at $(date -Is); stopping worker" >> "$LOG"
    kill -TERM "$PY_PID" 2>/dev/null
    wait "$PY_PID" 2>/dev/null
    if [ -f "$DONE_MARKER" ]; then
        echo "[chain] k=12 step complete, not resubmitting" >> "$LOG"
    elif [ "$LINK" -ge "$MAX_LINKS" ]; then
        echo "[chain] link cap $MAX_LINKS reached, not resubmitting" >> "$LOG"
    else
        echo "[chain] submitting link $((LINK + 1))" >> "$LOG"
        sbatch --export=ALL,QD_CHAIN_LINK=$((LINK + 1)) "$SCRIPT" >> "$LOG" 2>&1
    fi
    exit 0
}
trap requeue USR1

python -u "$ROOT/tools/run_molecular_growth.py" \
  --pack-dir "$PACK" \
  --growth   "$PACK/growth.yaml" \
  --parents  /scratch/iinfante/CP2K/Nucleation/xtb/CdSe/new_generator/runs/gxtb_cdse_target_k1k2_p1p5 \
  --k-from 1 --k-to 13 --p-parents all \
  --output   "$OUT" \
  >> "$LOG" 2>&1 &
PY_PID=$!
wait "$PY_PID"
RC=$?

echo "[chain] worker exit $RC  $(date -Is)" >> "$LOG"
if [ -f "$DONE_MARKER" ]; then
    echo "[chain] reached k=13, chain finished" >> "$LOG"
fi
exit $RC
