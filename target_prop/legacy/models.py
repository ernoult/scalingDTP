import numpy as np
import torch
import torch.nn as nn


class layer_fc(nn.Module):
    """
    Defines a fully connected autoencoder
    with feedforward and feedback weights
    """

    def __init__(self, in_size, out_size, iter, noise):
        super(layer_fc, self).__init__()
        self.f = nn.Linear(in_size, out_size)
        self.b = nn.Linear(out_size, in_size)

        # Number of feedback weights training iterations *per batch iteration*
        self.iter = iter

        # Standard deviation of the noise injected for feedback weights training
        self.noise = noise

    def ff(self, x):
        """
        Feedforward operator (x --> y = F(x))
        """

        x_flat = x.view(x.size(0), -1)
        y = self.f(x_flat)
        return y

    def bb(self, x, y):
        """
        Feedback operator (y --> r = G(y))
        """

        r = self.b(y)
        r = r.view(x.size())

        return r

    def forward(self, x, back=False):
        y = self.ff(x)

        if back:
            r = self.bb(x, y)
            return y, r
        else:
            return y

    def weight_b_sym(self):
        """
        Equalizes feedforward and feedback weights (useful for sanity checks)
        """

        with torch.no_grad():
            self.f.weight.data = self.b.weight.data.t()

    def compute_dist_angle(self, *args):
        """
        Computes angle and distance between feedforward and feedback weights
        """

        F = self.f.weight
        G = self.b.weight.t()

        dist = torch.sqrt(((F - G) ** 2).sum() / (F ** 2).sum())

        F_flat = torch.reshape(F, (F.size(0), -1))
        G_flat = torch.reshape(G, (G.size(0), -1))
        cos_angle = ((F_flat * G_flat).sum(1)) / torch.sqrt(
            ((F_flat ** 2).sum(1)) * ((G_flat ** 2).sum(1))
        )
        angle = (180.0 / np.pi) * (torch.acos(cos_angle).mean().item())

        return dist, angle

    def weight_b_train(self, input, optimizer, arg_return=False):
        """
        Trains feedback weights
        """

        nb_iter = self.iter
        sigma = self.noise

        for iter in range(1, nb_iter + 1):
            # 1- Compute y = F(input) and r=G(y)
            y, r = self(input, back=True)

            # 2- Perturbate x <-- x + noise and redo x--> y --> r
            noise = sigma * torch.randn_like(input)
            _, r_noise = self(input + noise, back=True)
            dr = r_noise - r

            # 3- Perturbate y <-- y + noise and redo y --> r
            noise_y = sigma * torch.randn_like(y)
            r_noise_y = self.bb(input, y + noise_y)
            dr_y = r_noise_y - r

            # 4- Compute the loss
            loss_b = (
                -2 * (noise * dr).view(dr.size(0), -1).sum(1).mean()
                + (dr_y ** 2).view(dr_y.size(0), -1).sum(1).mean()
            )
            optimizer.zero_grad()

            # 5- Update the feedback weights
            loss_b.backward()
            optimizer.step()

        if arg_return:
            return loss_b

    def weight_f_train(self, y, t, optimizer):
        """
        Trains the feedforward weights from
        feedforward prediction (y) and associated target (t)
        """

        # 1- Compute MSE between feedforward prediction and associated target
        loss_f = 0.5 * ((y - t) ** 2).view(y.size(0), -1).sum(1)
        loss_f = loss_f.mean()
        optimizer.zero_grad()

        # 2- Update feedforward weights
        loss_f.backward(retain_graph=True)
        optimizer.step()

        return loss_f

    def propagateError(self, r, t):
        """
        Computes G(t) - G(r)
        (error signal for the previous layer)
        """

        delta = self.bb(r, t) - r
        return delta


class layer_convpool(nn.Module):
    """
    Defines a fully connected autoencoder
    with feedforward and feedback weights
    """

    def __init__(self, args, in_channels, out_channels, activation, iter=None, noise=None):
        super(layer_convpool, self).__init__()
        self.f = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)

        # NOTE 1: the feedback operator is optional (e.g. the first conv layer doesn't need one)
        if iter is not None:
            self.b = nn.ConvTranspose2d(out_channels, in_channels, 3, stride=1, padding=1)
            self.unpool = nn.MaxUnpool2d(2, stride=2)

        if activation == "elu":
            self.rho = nn.ELU()
        elif activation == "relu":
            self.rho = nn.ReLU()

        self.iter = iter
        self.noise = noise

    def ff(self, x, ret_ind=False):
        """
        Feedforward operator (x --> y = F(x))
        """
        # conv -> rho -> pool
        y, ind = self.pool(self.rho(self.f(x)))

        if ret_ind:
            return y, ind
        else:
            return y

    def bb(self, x, y, ind):
        """
        Feedback operator (y --> r = G(y))
        """
        # unpool -> rho -> conv_transpose
        if hasattr(self, "b"):
            r = self.unpool(y, ind, output_size=x.size())
            r = self.b(self.rho(r), output_size=x.size())
            return r
        else:
            return None

    def forward(self, x, back=False):
        if back:
            y, ind = self.ff(x, ret_ind=True)
            r = self.bb(x, y, ind)
            return y, (r, ind)
        else:
            y = self.ff(x)
            return y

    def weight_b_sym(self):
        """
        Equalizes feedforward and feedback weights (useful for sanity checks)
        """

        with torch.no_grad():
            self.b.weight.data = self.f.weight.data

    def compute_dist_angle(self, *args):
        """
        Computes distance and angle between feedforward
        and feedback convolutional kernels
        """

        F = self.f.weight
        G = self.b.weight
        dist = torch.sqrt(((F - G) ** 2).sum() / (F ** 2).sum())

        F_flat = torch.reshape(F, (F.size(0), -1))
        G_flat = torch.reshape(G, (G.size(0), -1))
        cos_angle = ((F_flat * G_flat).sum(1)) / torch.sqrt(
            ((F_flat ** 2).sum(1)) * ((G_flat ** 2).sum(1))
        )
        angle = (180.0 / np.pi) * (torch.acos(cos_angle).mean().item())

        return dist, angle

    def weight_b_train(self, input, optimizer, arg_return=False):
        """
        Trains feedback weights
        """

        nb_iter = self.iter
        sigma = self.noise

        for iter in range(1, nb_iter + 1):

            # 1- Compute y = F(input) and r=G(y)
            y, (r, ind) = self(input, back=True)
            noise = sigma * torch.randn_like(input)

            # 2- Perturbate x <-- x + noise and redo x--> y --> r
            _, (r_noise, ind_noise) = self(input + noise, back=True)
            dr = r_noise - r

            # 3- Perturbate y <-- y + noise and redo y --> r
            noise_y = sigma * torch.randn_like(y)
            r_noise_y = self.bb(input, y + noise_y, ind)
            dr_y = r_noise_y - r

            # 4- Compute the loss
            loss_b = (
                -2 * (noise * dr).view(dr.size(0), -1).sum(1).mean()
                + (dr_y ** 2).view(dr_y.size(0), -1).sum(1).mean()
            )

            # 5- Update the weights
            optimizer.zero_grad()
            loss_b.backward()
            optimizer.step()

        if arg_return:
            return loss_b

    def weight_f_train(self, y, t, optimizer):
        """
        Trains forward weights
        """

        # 1- Compute MSE between feedforward prediction and associated target
        loss_f = 0.5 * ((y - t) ** 2).view(y.size(0), -1).sum(1)
        loss_f = loss_f.mean()
        optimizer.zero_grad()

        # 2- Update forward weights
        loss_f.backward(retain_graph=True)
        optimizer.step()

        return loss_f

    def propagateError(self, r_tab, t):
        """
        Computes G(t) - G(r)
        (error signal for the previous layer)
        """

        r = r_tab[0]
        ind = r_tab[1]
        delta = self.bb(r_tab[0], t, r_tab[1]) - r_tab[0]

        return delta


class VGG(nn.Module):
    """
    Defines a VGG-like architecture made up
    of fully connected and convolutional autoencoders
    """

    def __init__(self, args):
        super(VGG, self).__init__()

        # CIFAR-10 input dimensions
        size = 32
        args.C = [3] + args.C

        layers = nn.ModuleList([])

        # Build the first convolutional layer (without feedback operator G)
        layers.append(layer_convpool(args, args.C[0], args.C[1], args.activation))
        size = int(np.floor(size / 2))

        # Build the convolutional autoencoders
        for i in range(len(args.C) - 2):
            layers.append(
                layer_convpool(
                    args, args.C[i + 1], args.C[i + 2], args.activation, args.iter[i], args.noise[i]
                )
            )

            size = int(np.floor(size / 2))

        # Build the last (fully connected) autoencoder
        layers.append(layer_fc((size ** 2) * args.C[-1], 10, args.iter[-1], args.noise[-1]))

        self.layers = layers
        self.logsoft = nn.LogSoftmax(dim=1)
        self.noise = args.noise
        self.beta = args.beta

    def forward(self, x, ind_layer=None):
        """
        NOTE: it is *crucial* to detach nodes in the computational graph
        at the right places to ensure we compute gradients *locally*
        (both for the training of the feedback *and* of the feedforward
        weights)
        """

        s = x

        if ind_layer is None:
            for i in range(len(self.layers)):
                s = self.layers[i](s)

            return s

        else:
            for i in range(ind_layer):
                if i == ind_layer - 1:
                    s = s.detach()
                    s.requires_grad = True
                    s, r = self.layers[i](s, back=True)
                else:
                    s = self.layers[i](s)

            return s, r

    def weight_b_sym(self):
        """
        Equalizes feedforward and feedback weights (useful for sanity checks)
        """
        for i in range(len(self.layers)):
            self.layers[i].weight_b_sym()
