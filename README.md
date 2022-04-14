# Towards Scaling Difference Target Propagation with Backprop Targets

![](dtp_cartoon.png)

This repository is the official implementation of "Towards Scaling Difference Target Propagation with Backprop Targets", currently under review at ICML 2022. The following code runs on Python > 3.7 with Pytorch >= 1.7.0.
## Installation

```console
pip install -e .
```
(Optional): We suggest you use a conda environment. The specs of our environment are stored in `conda_env_specs.txt`.


## Naming of methods:


| Name in paper                       | Name in codebase                                                                                                                                                |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| L-DRL                               | [DTP](target_prop/models/dtp.py)                                                                                                                                |
| Backpropagation                     | [BaselineModel](target_prop/models/baseline.py)                                                                                                                 |
| DRL                                 | [meulemans_dtp](meulemans_dtp/README.md) (Based on [the original authors' repo](https://github.com/meulemansalex/theoretical_framework_for_target_propagation)) |
| Target Propagation                  | [TargetProp](target_prop/models/tp.py)                                                                                                                          |
| Difference Target Propagation       | [VanillaDTP](target_prop/models/vanilla_dtp.py)                                                                                                                 |
| "Parallel" L-DRL (not in the paper) | [ParallelDTP](target_prop/models/parallel_dtp.py)                                                                                                               |


## Codebase structure
The main logic of our method is in [target_prop/models/dtp.py](target_prop/models/dtp.py)

An initial PyTorch implementation of our DTP model can be found under [target_prop/legacy](target_prop/legacy).
This model was then re-implemented using [PyTorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning).


Here is how the codebase is roughly structured:
```
├── main.py                # training script (legacy)
├── meulemans_dtp          # Codebase for DRL (Meulemans repo)
├── numerical_experiments  # Initial scripts for creating the figures (used for fig. 4.2) 
└── target_prop
    ├── datasets  # Datasets
    ├── legacy    # initial implementation
    ├── models    # Code for all the models except DRL
    └── networks  # Networks (SimpleVGG, LetNet, ResNet)
```

## Running the code

- Recreating figure 4.2:
  ```console
  $ python -m numerical_experiments figure_4_2
  ```
  The figure save location will then be displayed on the console. 

- Recreating figure 4.3:
  ```console
  $ pytest -vv target_prop/networks/lenet_test.py
  $ python target_prop/legacy/plot.py
  ```

To see a list of available command-line options, use the "--help" command.
```console
python main.py --help
```

To run the pytorch-lightning re-implementation of DTP on CIFAR-10, use the following command:
```console
python main.py model=dtp dataset=cifar10
```

To use the modified version of the above DTP model, with parallel feedback weight training on CIFAR-10, use the following command:
```console
python main.py model=parallel_dtp dataset=cifar10
```

To run backprop baseline, do:
```console
python main.py model=backprop dataset=cifar10
```



### ImageNet

To train with DTP on downsampled ImageNet 32x32 dataset, do:
```console
python main.py model=dtp dataset=imagenet32
```


### Legacy Implementation
To check training on CIFAR-10, type the following command in the terminal:

```console
python main_legacy.py --batch-size 128 \
    --C 128 128 256 256 512 \
    --iter 20 30 35 55 20 \
    --epochs 90 \
    --lr_b 1e-4 3.5e-4 8e-3 8e-3 0.18 \
    --noise 0.4 0.4 0.2 0.2 0.08 \
    --lr_f 0.08 \
    --beta 0.7 \
    --path CIFAR-10 \
    --scheduler --wdecay 1e-4
```
