#!/bin/bash
#SBATCH -J qd_motifs_k1k8
#SBATCH -t 72:00:00
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem-per-cpu=2g
#SBATCH --qos=regular

# Resume growth_lattice_motifs_k1_to_k8 after a kill mid k=7.
#
# --k-from is the *parent* k.  Do NOT use --k-from 7: that starts k=8 and
# skips the unfinished k=7 p=10 bin.  Same command as the original job:
#   --k-from 1 --k-to 8 --output <same dir>  (resume is on by default)
# Finished steps (.step_k001_complete … k005) are skipped.  Move Z bins
# already in zb_motifs.jsonl (k=7 p=1…9 if those lines exist) are skipped.
# Incomplete bins (k=7 p=10) are opted again.
#
# On HPC, git pull nucleation (needs this resume skip + STAGE 2 cap) and
# keep the same growth.yaml already in $PACK (zb_motif_log / zb_channel_b).
#
# Append the wrapper log so the previous stdout is not truncated:
#   >> "$PACK/logs/….log"

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
OUT=$PACK/growth_lattice_motifs_k1_to_k8

mkdir -p "$PACK/logs"
cd "$ROOT"

python -u "$ROOT/tools/run_molecular_growth.py" \
  --pack-dir "$PACK" \
  --growth   "$PACK/growth.yaml" \
  --parents  /scratch/iinfante/CP2K/Nucleation/xtb/CdSe/new_generator/runs/gxtb_cdse_target_k1k2_p1p5 \
  --k-from 1 --k-to 8 --p-parents all \
  --output   "$OUT" \
  >> "$PACK/logs/$(basename $OUT).log" 2>&1
