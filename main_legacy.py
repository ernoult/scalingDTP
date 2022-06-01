# coding=utf-8
import argparse
import os
import pickle
import sys

import torch
from tqdm import tqdm

from target_prop.legacy import (
    VGG,
    createDataset,
    createHyperparameterfile,
    createOptimizers,
    createPath,
    test,
    train_batch,
)


def main():
    parser = argparse.ArgumentParser(description="Testing idea of Yoshua")

    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="number of epochs to train feedback weights(default: 15)",
    )
    parser.add_argument(
        "--iter",
        nargs="+",
        type=int,
        default=[5, 10],
        help="number of learning iterating of feedback weights layer-wise per batch (default: [5, 10])",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="batch dimension (default: 128)"
    )
    parser.add_argument("--device-label", type=int, default=0, help="device (default: 1)")
    parser.add_argument("--lr_f", type=float, default=0.05, help="learning rate (default: 0.05)")
    parser.add_argument(
        "--lr_b",
        nargs="+",
        type=float,
        default=[0.05, 0.05],
        help="learning rates for the feedback weights (default: [0.05, 0.05])",
    )
    parser.add_argument("--beta", type=float, default=0.1, help="nudging parameter (default: 0.1)")
    parser.add_argument(
        "--C", nargs="+", type=int, default=[32, 64], help="tab of channels (default: [32, 64])"
    )
    parser.add_argument(
        "--noise",
        nargs="+",
        type=float,
        default=[0.05, 0.5],
        help="tab of noise amplitude (default: [0.05, 0.5])",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="elu",
        help="activation function in conv layers (default: elu)",
    )
    parser.add_argument(
        "--path", type=str, default=None, help="Path directory for the results (default: None)"
    )
    parser.add_argument(
        "--last-trial",
        default=False,
        action="store_true",
        help="specifies if the current trial is the last one (default: False)",
    )
    parser.add_argument("--seed", type=int, default=None, help="seed selected (default: None)")
    parser.add_argument(
        "--scheduler",
        default=False,
        action="store_true",
        help="use of a learning rate scheduler for the forward weights (default: False)",
    )
    parser.add_argument("--wdecay", type=float, default=None, help="Weight decay (default: None)")

    args = parser.parse_args()

    train_loader, test_loader = createDataset(args)

    if args.device_label >= 0:
        device = torch.device("cuda:" + str(args.device_label))
    else:
        device = torch.device("cpu")

    if args.seed is not None:
        seed = args.seed
        torch.manual_seed(seed)
    else:
        g = torch.Generator(device=device)
        seed = g.seed()
        torch.manual_seed(seed)

    print("Selected seed: {}".format(seed))

    # Create a directory to save results and hyperparameters
    if args.path is not None:
        BASE_PATH = createPath(args)
        command_line = " ".join(sys.argv)
        createHyperparameterfile(BASE_PATH, command_line, seed, args)

    # Create neural network
    net = VGG(args)
    net = net.to(device)
    print(net)

    # Create optimizers for forward and feedback weights
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    optimizers = createOptimizers(net, args, forward=True)

    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[0], 85, eta_min=1e-5)
        print("We are using a learning rate scheduler!")

    train_acc = []
    test_acc = []

    # train the neural network by DTP
    for epoch in range(args.epochs):
        train_loss = 0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")

        for batch_idx, (data, target) in enumerate(pbar):
            net.train()
            data, target = data.to(device), target.to(device)

            # compute DTP gradient on the current batch
            pred, loss, layer_losses_b, layer_losses_f = train_batch(
                args, net, data, optimizers, target, criterion
            )

            train_loss += loss.item()
            _, predicted = pred.max(1)
            targets = target
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            train_accuracy = correct / total
            loss_f = sum(layer_losses_f) / len(layer_losses_f)
            loss_b = sum(layer_losses_b) / len(layer_losses_b)

            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.3f}",
                    "Train Acc": f"{train_accuracy:.2%}",
                    "F_loss": f"{loss_f:.3f}",
                    "B_loss": f"{loss_b:.3f}",
                }
            )
            # progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Train Acc: %.3f%% (%d/%d)'% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        train_acc.append(100.0 * correct / total)
        test_acc_temp = test(net, test_loader, device)
        test_acc.append(test_acc_temp)

        # save accuracies in the results directory
        if args.path is not None:
            results = {"train_acc": train_acc, "test_acc": test_acc}
            outfile = open(os.path.join(BASE_PATH, "results"), "wb")
            pickle.dump(results, outfile)
            outfile.close()

        if args.scheduler:
            scheduler.step()

        # if the train accuracy is less than 30%, kill training
        if train_accuracy < 0.30:
            print(f"Accuracy is terrible ({train_accuracy:.2%}), exiting early")
            break


if __name__ == "__main__":
    main()
