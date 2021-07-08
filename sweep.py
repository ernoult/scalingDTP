""" Runs a stupid simple sweep with random search. """
# TODO: Use `orion` or wandb sweeps instead.

from main import main
n_runs = 10
for run in range(n_runs):
    main(sample_hparams=True)
