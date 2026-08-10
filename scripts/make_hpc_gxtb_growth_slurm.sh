#!/bin/bash
# Generate Slurm scripts for (1) g-xTB molecular *map* bins and
# (2) molecular *growth* steps on HPC.
#
# Usage (from QD_Builder root on the cluster, after editing paths below):
#   bash scripts/make_hpc_gxtb_growth_slurm.sh
#   sbatch slurm_jobs/run_gxtb_k2_p1p5.slurm
#   sbatch slurm_jobs/run_growth_k2_to_k3.slurm
#
# Only g-xTB is generated (no plain GFN-xTB twin). Growth jobs depend on
# finished parent map directories (index.csv + k***/p***/*_xtb.xyz).

set -euo pipefail

# ===================== edit these for your HPC =====================
CONDA_PROFILE="/scicomp/builds/Rocky/8.7/Common/software/Mamba/23.11.0-0/etc/profile.d/conda.sh"
CONDA_ENV="qd-builder"
QD_ROOT="/scratch/iinfante/escience/QD_Builder"
# Pack with run_gxtb.yaml + growth.yaml (copy of graphs/growth on login node)
PACK_DIR="/scratch/iinfante/escience/Nucleation/graphs/growth"
# Parent map runs live here
RUNS_ROOT="/scratch/iinfante/escience/Nucleation/graphs/runs"
GXTB_PATH="/scratch/iinfante/xtb/xtb-6.7.1"   # or your g-xTB install
MAP_PY="${QD_ROOT}/tools/run_molecular_map.py"
GROWTH_PY="${QD_ROOT}/tools/run_molecular_growth.py"
CPUS=8
TIME="24:00:00"
QOS="regular"
# ===================================================================

mkdir -p slurm_jobs logs

# -------------------- g-xTB map intervals ---------------------------
# "kmin kmax pmin pmax tag"
MAP_INTERVALS=(
  "1 2 1 5 k1k2_p1p5"
  # "3 3 1 4 k3_p1p4"
  # "4 4 1 4 k4_p1p4"
)

for item in "${MAP_INTERVALS[@]}"; do
  read -r kmin kmax pmin pmax tag <<< "$item"
  ARGS="--kmin ${kmin} --kmax ${kmax} --pmin ${pmin} --pmax ${pmax}"
  OUT_RUN="${RUNS_ROOT}/gxtb_cdse_${tag}"
  SLURM="slurm_jobs/run_gxtb_${tag}.slurm"
  cat > "$SLURM" << EOF
#!/bin/bash
#SBATCH -J gxtb_${tag}
#SBATCH -t ${TIME}
#SBATCH -n 1
#SBATCH -c ${CPUS}
#SBATCH --mem-per-cpu=2g
#SBATCH --qos=${QOS}

export PYTHONUNBUFFERED=1
source ${CONDA_PROFILE}
mamba activate ${CONDA_ENV}

export PATH="${GXTB_PATH}/bin:\$PATH"
export XTBPATH="${GXTB_PATH}/share/xtb"
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_STACKSIZE=1G
ulimit -s unlimited

cd ${QD_ROOT}
mkdir -p logs ${OUT_RUN}

# Map driver: pack-dir style — use absolute path to run_gxtb.yaml in PACK_DIR
python -u ${MAP_PY} \\
  --yaml ${PACK_DIR}/run_gxtb.yaml \\
  ${ARGS} \\
  --output ${OUT_RUN} \\
  > logs/gxtb_${tag}.log 2>&1
EOF
  echo "Generated $SLURM  →  ${OUT_RUN}"
done

# -------------------- growth steps (g-xTB decorate+opt) -------------
# "k_from p_parents_csv parents_tag out_tag"
# parents_tag selects RUNS_ROOT/gxtb_cdse_${parents_tag}
GROWTH_STEPS=(
  "2 2,3 k1k2_p1p5 growth_k2_to_k3"
  # "3 all growth_k2_to_k3 growth_k3_to_k4"
)

for item in "${GROWTH_STEPS[@]}"; do
  read -r kfrom pparents ptag otag <<< "$item"
  PARENTS="${RUNS_ROOT}/gxtb_cdse_${ptag}"
  OUT_RUN="${RUNS_ROOT}/${otag}"
  PARG=""
  if [ "$pparents" != "all" ]; then
    PARG="--p-parents ${pparents}"
  fi
  SLURM="slurm_jobs/run_growth_${otag}.slurm"
  cat > "$SLURM" << EOF
#!/bin/bash
#SBATCH -J grow_${otag}
#SBATCH -t ${TIME}
#SBATCH -n 1
#SBATCH -c ${CPUS}
#SBATCH --mem-per-cpu=2g
#SBATCH --qos=${QOS}

export PYTHONUNBUFFERED=1
source ${CONDA_PROFILE}
mamba activate ${CONDA_ENV}

export PATH="${GXTB_PATH}/bin:\$PATH"
export XTBPATH="${GXTB_PATH}/share/xtb"
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_STACKSIZE=1G
ulimit -s unlimited

cd ${QD_ROOT}
mkdir -p logs ${OUT_RUN}

python -u ${GROWTH_PY} \\
  --pack-dir ${PACK_DIR} \\
  --parents  ${PARENTS} \\
  --k-from   ${kfrom} \\
  ${PARG} \\
  --output   ${OUT_RUN} \\
  > logs/growth_${otag}.log 2>&1
EOF
  echo "Generated $SLURM  →  parents=${PARENTS}  out=${OUT_RUN}"
done

echo ""
echo "Slurm scripts in slurm_jobs/"
echo "Submit map first, then growth when parents exist, e.g.:"
echo "  sbatch slurm_jobs/run_gxtb_k1k2_p1p5.slurm"
echo "  sbatch slurm_jobs/run_growth_growth_k2_to_k3.slurm"
