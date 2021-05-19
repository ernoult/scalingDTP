## Revisiting Difference Target Propagation

To check training on CIFAR-10, type the following command in the terminal:

```
python main.py --batch-size 128 --C 128 128 256 256 512 --iter 20 30 35 55 20 --epochs 90 --lr_b 1e-4 3.5e-4 8e-3 8e-3 0.18 --noise 0.4 0.4 0.2 0.2 0.08 --lr_f 0.08 --beta 0.7 --path CIFAR-10 --scheduler --wdecay 1e-4
```
