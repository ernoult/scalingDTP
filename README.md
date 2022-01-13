# Revisiting Difference Target Propagation

The following code runs on Python > 3.6 with Pytorch 1.7.0.
## Installation
```console
pip install -e .
```

## Running the code
To run the pytorch-lightning re-implementation of DTP on CIFAR-10, use the following command:
```console
python main_pl.py run dtp simple_vgg
```

To use the modified version of the above DTP model, with "parallel" feedback weight training on CIFAR-10, use the following command:
```console
python main_pl.py run parallel_dtp simple_vgg
```

### ImageNet

To train with DTP on downsampled ImageNet 32x32 dataset, do:
```
<<<<<<< HEAD
python main_pl.py dtp --dataset imagenet32
=======
python main_pl.py run dtp <architecture> --dataset imagenet32
>>>>>>> master
```


### Legacy Implementation
To check training on CIFAR-10, type the following command in the terminal:

```console
python main.py --batch-size 128 \
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
