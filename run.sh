#!/bin/bash
#SBATCH --array=0-5%2
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=10GB
conda activate target_prop

cd ~/target_prop/scalingDTP

python main_pl.py dtp --data_dir ~/scratch --seed $SLURM_ARRAY_TASK_ID

