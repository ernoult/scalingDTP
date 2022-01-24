"""
This script iterates over all combinations of hyper-parameters provided in sweep config 
and submits batch job for each experiment to ComputeCanada cluster.
"""
import itertools
import subprocess

if __name__ == "__main__":
    # Meta info
    account = "def-eugenium"
    user = "amoudgl"
    network = "simple_vgg"
    algo = "backprop"
    time = "5:0:0"

    # Hyper params to sweep over
    sweep_config = [
        [            
            "--batch_size 256",
        ],
        [
            "--num_workers 4",
        ],        
        [
            "--type sgd --lr 0.01 --use_scheduler true",
            "--type sgd --lr 0.05 --use_scheduler true",
            # "--b_optim.type adam --b_optim.lr 1e-4 3.5e-4 8e-3 8e-3 1e-2",
        ],
        [
            "--dataset imagenet32",
        ],
        [
            "step",
        ],
        [
            "--step_size 30",
            "--step_size 45",
        ],
        [
            "--seed 124",
            "--seed 125",
            "--seed 126",
            "--seed 127",
        ]
    ]
    init_commands = f"module load python/3.8 && source /scratch/{user}/py38/bin/activate && cd /scratch/{user}/scalingDTP && export WANDB_MODE=offline"
    python_command = f"python main_pl.py run {algo} {network}"
    sbatch_command = f"sbatch --gres=gpu:1 --account={account} --cpus-per-task=16 --time={time} --mem=48G"

    # Submit batch jobs for all combinations
    all_args = list(itertools.product(*sweep_config))
    print(f"Total jobs = {len(all_args)}")
    for args in all_args:
        args = " ".join(args)
        job_command = (
            sbatch_command
            + ' --wrap="'
            + init_commands
            + " && "
            + python_command
            + " "
            + args
            + '"'
        )
        print(job_command)
        subprocess.run(job_command, shell=True)

# sbatch --gres=gpu:1 --account=def-eugenium --time=10:0:0 --mem=48G --wrap="module load python/3.8 && source /scratch/amoudgl/py38/bin/activate && cd /scratch/amoudgl/scalingDTP && python main_pl.py run dtp simple_vgg --dataset cifar10 --f_optim.lr 0.02 --num_workers 2 cosine"
# sbatch --gres=gpu:1 --account=def-eugenium --time=11:0:0 --cpus-per-task=16 --mem=48G --wrap="module load python/3.8 && source /scratch/amoudgl/py38/bin/activate && cd /scratch/amoudgl/scalingDTP && export WANDB_MODE=offline && python main_pl.py run dtp simple_vgg --dataset cifar10 --f_optim.lr 0.02 --num_workers 2 cosine"