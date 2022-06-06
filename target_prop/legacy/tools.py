import copy
import datetime
import os
import stat
from shutil import copyfile

import numpy as np
import torch
import torch.nn.functional as F
import torchvision


def train_batch(args, net, data, optimizers, target=None, criterion=None, **kwargs):
    """
    Function which computes the parameter gradients given by DTP on a given batch
    NOTE: it is the central function of this whole code!
    """

    optimizer_f = optimizers[0]

    # ****FEEDBACK WEIGHTS TRAINING****#
    pred, layer_losses_b = train_backward(net, data)

    # *********FORWARD WEIGHTS TRAINING********#
    loss, layer_losses_f = train_forward(net, data, target, criterion, optimizer_f, args)
    return pred, loss, layer_losses_b, layer_losses_f


def train_backward(net, data):
    # 1- Compute the first hidden layer and detach the resulting node
    y = net.layers[0](data).detach()

    # 2- Layer-wise autoencoder training begins:
    losses = []
    for id_layer in range(len(net.layers) - 1):
        # 3- Train the current autoencoder (NOTE: there is no feedback operator paired to the first layer net.layers[0])
        loss_b = net.layers[id_layer + 1].weight_b_train(y, True)
        losses.append(loss_b)

        # 4- Compute the next hidden layer
        y = net.layers[id_layer + 1](y).detach()

    pred = torch.exp(net.logsoft(y))
    return pred, losses


def train_forward(net, data, target, criterion, optimizer_f, args):
    # 1- Compute the output layer (y) and the reconstruction of the penultimate layer (r = G(y))
    """
    NOTE 1: the flag ind_layer specifies where the forward pass stops (default: None)
    NOTE 2: if ind_layer=n, layer n-1 is detached from the computational graph
    """
    optimizer_f.zero_grad()
    y, r = net(data, ind_layer=len(net.layers))

    # 2- Compute the loss
    L = criterion(y.float(), target).squeeze()

    # 3- Compute the first target
    init_grads = torch.tensor(
        [1 for i in range(y.size(0))], dtype=torch.float, device=y.device, requires_grad=True
    )
    grads = torch.autograd.grad(L, y, grad_outputs=init_grads, create_graph=True)
    delta = -args.beta * grads[0]
    t = (y + delta).detach()

    # 4- Layer-wise feedforward training begins
    layer_losses = [torch.tensor(0.0)] * len(net.layers)

    for id_layer in range(len(net.layers)):
        # 5- Train current forward weights so that current layer matches its target
        loss_f = net.layers[-1 - id_layer].weight_f_train(y, t, optimizer_f)
        layer_losses[-1 - id_layer] = loss_f

        # 6- Compute the previous target (except if we already have reached the first hidden layer)
        if id_layer < len(net.layers) - 1:
            # 7- Compute delta^n = G(s^{n+1} + t^{n+1}) - G(s^{n+1})
            delta = net.layers[-1 - id_layer].propagateError(r, t)

            # 8- Compute the feedforward prediction s^n and r=G(s^n)
            y, r = net(data, ind_layer=len(net.layers) - 1 - id_layer)

            # 9- Compute the target t^n= s^n + delta^n
            t = (y + delta).detach()

        if id_layer == 0:
            loss = loss_f

    optimizer_f.step()
    return loss, layer_losses


def createOptimizers(net, args, forward=False):

    """
    Function which initializes the optimizers of
    the feedforward and feedback weights
    """

    for i in range(len(net.layers) - 1):
        optim_params_b = [{"params": net.layers[i + 1].b.parameters(), "lr": args.lr_b[i]}]
        net.layers[i + 1].optimizer = torch.optim.SGD(optim_params_b, momentum=0.9)

    if forward:
        optim_params_f = []

        if args.wdecay is None:
            for i in range(len(net.layers)):
                optim_params_f.append({"params": net.layers[i].f.parameters(), "lr": args.lr_f})
        else:
            for i in range(len(net.layers)):
                optim_params_f.append(
                    {
                        "params": net.layers[i].f.parameters(),
                        "lr": args.lr_f,
                        "weight_decay": args.wdecay,
                    }
                )
            print("We are using weight decay!")

        optimizer_f = torch.optim.SGD(optim_params_f, momentum=0.9)
        return [optimizer_f]
    else:
        return [None]


def test(net, test_loader, device):

    """
    Function to evaluate the neural network on the test set
    """

    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            y = net(data)
            pred = torch.exp(net.logsoft(y))
            target = F.one_hot(target, num_classes=10).float()
            _, predicted = pred.max(1)
            _, targets = target.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_acc = 100.0 * correct / total

    return test_acc


def copy(y, ind_y):
    y_copy = []

    for i in range(len(y)):
        y_copy.append(y[i].clone())

    # WATCH OUT: detach previous node!
    y_copy[ind_y - 1] = y_copy[ind_y - 1].detach()

    return y_copy


# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)

# TOTAL_BAR_LENGTH = 65.
# last_time = time.time()
# begin_time = last_time

# def progress_bar(current, total, msg=None):
#     '''
#     Function for the progression bar taken from another repo)
#     '''

#     global last_time, begin_time
#     if current == 0:
#         begin_time = time.time()  # Reset for new bar.

#     cur_len = int(TOTAL_BAR_LENGTH*current/total)
#     rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

#     sys.stdout.write(' [')
#     for i in range(cur_len):
#         sys.stdout.write('=')
#     sys.stdout.write('>')
#     for i in range(rest_len):
#         sys.stdout.write('.')
#     sys.stdout.write(']')

#     cur_time = time.time()
#     step_time = cur_time - last_time
#     last_time = cur_time
#     tot_time = cur_time - begin_time

#     L = []
#     L.append('  Step: %s' % format_time(step_time))
#     L.append(' | Tot: %s' % format_time(tot_time))
#     if msg:
#         L.append(' | ' + msg)

#     msg = ''.join(L)
#     sys.stdout.write(msg)
#     for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
#         sys.stdout.write(' ')

#     # Go back to the center of the bar.
#     for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
#         sys.stdout.write('\b')
#     sys.stdout.write(' %d/%d ' % (current+1, total))

#     if current < total-1:
#         sys.stdout.write('\r')
#     else:
#         sys.stdout.write('\n')
#     sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ""
    i = 1
    if days > 0:
        f += str(days) + "D"
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + "h"
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + "m"
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + "s"
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + "ms"
        i += 1
    if f == "":
        f = "0ms"
    return f


def createPath(args):
    """
    Function which creates the folder of results
    """

    if args.path is None:
        BASE_PATH = os.getcwd() + "/train"
        if not os.path.exists(BASE_PATH):
            os.mkdir(BASE_PATH)
        BASE_PATH = BASE_PATH + "/" + datetime.datetime.now().strftime("%Y-%m-%d")

    else:
        BASE_PATH = os.getcwd() + "/" + args.path

    if not os.path.exists(BASE_PATH):
        os.mkdir(BASE_PATH)

    files = os.listdir(BASE_PATH)

    BASE_PATH_glob = BASE_PATH

    if not files:
        BASE_PATH = BASE_PATH + "/" + "Trial-1"
    else:
        tab = []
        for names in files:
            if not names[-2] == "-":
                tab.append(int(names[-2] + names[-1]))
            else:
                tab.append(int(names[-1]))

        BASE_PATH = BASE_PATH + "/" + "Trial-" + str(max(tab) + 1)

    os.mkdir(BASE_PATH)
    filename = "results"

    copyfile("plotFunctions.py", BASE_PATH + "/plotFunctions.py")

    if args.last_trial:
        copyfile("compute_stats.py", BASE_PATH_glob + "/compute_stats.py")

    return BASE_PATH


def createHyperparameterfile(BASE_PATH, command_line, seed, args):
    """
    Function which creates the .txt file containing hyperparameters
    """

    command = "python " + command_line

    if args.seed is None:
        command += " --seed " + str(seed)

    hyperparameters = open(BASE_PATH + r"/hyperparameters.txt", "w+")
    L = [
        "List of hyperparameters "
        + "("
        + datetime.datetime.now().strftime("cuda" + str(args.device_label) + "-%Y-%m-%d")
        + ") \n",
        "- number of channels per conv layer: {}".format(args.C) + "\n",
        "- number of feedback optim steps per batch: {}".format(args.iter) + "\n",
        "- activation function: {}".format(args.activation) + "\n",
        "- noise: {}".format(args.noise) + "\n",
        "- learning rate for forward weights: {}".format(args.lr_f) + "\n",
        "- learning rate for feedback weights: {}".format(args.lr_b) + "\n",
        "- batch size: {}".format(args.batch_size) + "\n",
        "- number of epochs: {}".format(args.epochs) + "\n",
        "- seed: {}".format(seed) + "\n",
        "\n To reproduce this simulation, type in the terminal:\n",
        "\n" + command + "\n",
    ]

    hyperparameters.writelines(L)
    hyperparameters.close()

    script = open(BASE_PATH + r"/reproduce_exp.sh", "w+")
    L = ["(cd " + os.getcwd() + ";" + command + ")"]
    script.writelines(L)
    script.close()

    script_name = BASE_PATH + "/reproduce_exp.sh"
    st = os.stat(script_name)
    os.chmod(script_name, st.st_mode | stat.S_IEXEC)


def createDataset(args):
    transform_train = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.RandomCrop(size=[32, 32], padding=4, padding_mode="edge"),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465), std=(3 * 0.2023, 3 * 0.1994, 3 * 0.2010)
            ),
        ]
    )

    transform_test = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465), std=(3 * 0.2023, 3 * 0.1994, 3 * 0.2010)
            ),
        ]
    )
    from pathlib import Path

    DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
    cifar10_train_dset = torchvision.datasets.CIFAR10(
        DATA_DIR, train=True, transform=transform_train, download=True
    )
    cifar10_test_dset = torchvision.datasets.CIFAR10(
        DATA_DIR, train=False, transform=transform_test, download=True
    )

    val_index = np.random.randint(10)
    val_samples = list(range(5000 * val_index, 5000 * (val_index + 1)))

    train_loader = torch.utils.data.DataLoader(
        cifar10_train_dset, batch_size=args.batch_size, shuffle=True, num_workers=1
    )
    test_loader = torch.utils.data.DataLoader(
        cifar10_test_dset, batch_size=200, shuffle=False, num_workers=1
    )

    return train_loader, test_loader
