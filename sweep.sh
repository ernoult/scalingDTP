#!/bin/bash
#SBATCH --array=0-20%5
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=10GB
module load anaconda/3
conda activate target_prop

cd ~/target_prop/scalingDTP

python main_pl.py sweep "$@" --data_dir $SLURM_TMPDIR
