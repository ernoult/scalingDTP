"""
This script iterates over all combinations of hyper-parameters provided in sweep config 
and submits batch job for each experiment to ComputeCanada cluster.
"""
import itertools
import subprocess

if __name__ == "__main__":
    # Meta info
    account = "rrg-ebrahimi"
    user = "muawizc"
    network = "simple_vgg"
    algo = "dtp"

    # Hyper params to sweep over
    sweep_config = [
        [            
            # "--batch_size 128",   # CIFAR10 default
            "--batch_size 256",
            "--batch_size 512",
        ],
        [
            "--feedback_training_iterations 20 30 35 55 20",  # CIFAR10 default
            "--feedback_training_iterations 25 35 40 60 25",
            "--feedback_training_iterations 35 45 50 70 35",
        ],
        [
            "--b_optim.type sgd --b_optim.momentum 0.9 --b_optim.lr 1e-4, 3.5e-4, 8e-3, 8e-3, 0.18",  # CIFAR10 default
            "--b_optim.type adam --b_optim.lr 1e-4, 3.5e-4, 8e-3, 8e-3, 1e-3",
            # "--b_optim.type adam --b_optim.lr 1e-4, 3.5e-4, 8e-3, 8e-3, 1e-2",
        ],
        [
            "f_optim.lr 0.08",  # CIFAR10 default
            "f_optim.lr 0.01",
            # "f_optim.lr 0.005",
            # "f_optim.lr 0.001",
        ],
        [
            "--dataset imagenet32",
            "--dataset imagenet32_3xstd",
        ],
        # TODO: Try sweeping over learning rate schedule
    ]
    init_commands = f"module load python/3.8 && source /scratch/{user}/py38/bin/activate && cd /scratch/{user}/scalingDTP && export WANDB_MODE=offline"
    python_command = f"python main_pl.py run {algo} {network}"
    sbatch_command = f"sbatch --gres=gpu:1 --account={account} --time=120:0:0 --mem=48G"

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
