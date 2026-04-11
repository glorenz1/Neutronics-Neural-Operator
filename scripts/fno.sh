#!/bin/bash
#SBATCH --job-name=fno
#SBATCH --account=isaac-utk0459
#SBATCH --partition=campus
#SBATCH --qos=campus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# =========================
# EDIT THESE PATHS
# =========================
#SBATCH --output=/path/to/your/results/slurm-%j.out
#SBATCH --error=/path/to/your/results/slurm-%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nno

# =========================
# EDIT THESE PATHS
# =========================
WORKDIR=/path/to/your/project
SAMPLES_DIR=/path/to/your/samples
RESULTS_DIR=/path/to/your/results

mkdir -p "${RESULTS_DIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

TRAIN_SCRIPT="${WORKDIR}/fno_train.py"
PLOT_SCRIPT="${WORKDIR}/fno_plot.py"
CHECKPOINT="${WORKDIR}/fno_trained.pt"

cd "${WORKDIR}"

echo "Starting training..."
python "${TRAIN_SCRIPT}" --data "${SAMPLES_DIR}"

echo "Training finished. Starting plotting..."
python "${PLOT_SCRIPT}" \
    --checkpoint "${CHECKPOINT}" \
    --data "${SAMPLES_DIR}" \
    --out-dir "${RESULTS_DIR}"
