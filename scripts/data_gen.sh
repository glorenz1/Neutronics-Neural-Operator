#!/bin/bash
#SBATCH --job-name=openmc
#SBATCH --account=isaac-utk0459
#SBATCH --partition=campus
#SBATCH --qos=campus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=08:00:00

# =========================
# EDIT THESE PATHS
# =========================
#SBATCH --output=/path/to/your/logs/launcher_%j.out
#SBATCH --error=/path/to/your/logs/launcher_%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nno

# =========================
# EDIT THESE PATHS
# =========================
WORKDIR=/path/to/your/project
OUTDIR=/path/to/your/samples
LOGDIR=/path/to/your/logs
OPENMC_XS=/path/to/your/cross_sections.xml

export OPENMC_CROSS_SECTIONS="${OPENMC_XS}"

SCRIPT="${WORKDIR}/openmc_data.py"
TIER=fast
TOTAL_SAMPLES=3000
CHUNK_SIZE=48
ARRAY_CONCURRENCY=48
SCRIPT_PATH=$0
SAMPLE_CPUS=4
SAMPLE_MEM=16GB
SAMPLE_TIME=02:00:00

mkdir -p "${OUTDIR}"
mkdir -p "${LOGDIR}"

submit_launcher() {
    local next_start=$1
    shift
    sbatch --parsable \
        "$@" \
        --job-name=openmc \
        --cpus-per-task=1 \
        --mem=1G \
        --time=08:00:00 \
        --output="${LOGDIR}/launcher_%j.out" \
        --error="${LOGDIR}/launcher_%j.err" \
        --export=ALL,MODE=launcher,NEXT_START="${next_start}" \
        "${SCRIPT_PATH}"
}

run_launcher() {
    local start=${NEXT_START:-0}
    local end=$((start + CHUNK_SIZE - 1))

    if (( start >= TOTAL_SAMPLES )); then
        echo "launcher complete"
        exit 0
    fi

    if (( end >= TOTAL_SAMPLES )); then
        end=$((TOTAL_SAMPLES - 1))
    fi

    local chunk_job_id
    chunk_job_id=$(
        sbatch --parsable \
            --job-name=openmc_fno \
            --cpus-per-task="${SAMPLE_CPUS}" \
            --mem="${SAMPLE_MEM}" \
            --time="${SAMPLE_TIME}" \
            --array="${start}-${end}%${ARRAY_CONCURRENCY}" \
            --output="${LOGDIR}/openmc_%A_%a.out" \
            --error="${LOGDIR}/openmc_%A_%a.err" \
            --export=ALL,MODE=sample \
            "${SCRIPT_PATH}"
    )
    echo "submitted chunk ${start}-${end} as job ${chunk_job_id}"

    start=$((end + 1))
    if (( start < TOTAL_SAMPLES )); then
        local next_launcher_job_id
        next_launcher_job_id=$(submit_launcher "${start}" --dependency="afterany:${chunk_job_id}")
        echo "submitted next launcher job ${next_launcher_job_id} for start ${start}"
    else
        echo "all chunks submitted"
    fi
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    launcher_job_id=$(submit_launcher 0)
    echo "submitted launcher job ${launcher_job_id}"
    exit 0
fi

if [[ "${MODE:-launcher}" == "launcher" ]]; then
    run_launcher
    exit 0
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "SLURM_ARRAY_TASK_ID is not set. Submit sample jobs with sbatch --array=..."
    exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

TASK_ID=${SLURM_ARRAY_TASK_ID}
SEED=$((0xC0FFEE + TASK_ID))
OUTFILE=$(printf "%s/sample_%04d.npz" "${OUTDIR}" "${TASK_ID}")

if [ -f "${OUTFILE}" ]; then
    echo "Sample ${TASK_ID} already exists, skipping."
    exit 0
fi

cd "${WORKDIR}"
python "${SCRIPT}" "${TASK_ID}" "${OUTDIR}" "${SEED}" "${TIER}"
