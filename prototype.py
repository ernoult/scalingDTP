# coding=utf-8
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import os
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


from plotFunctions import *
from tools import *

parser = argparse.ArgumentParser(description='Testing idea of Yoshua')


parser.add_argument('--in_size', type=int, default=784, help='input dimension (default: 784)')   
parser.add_argument('--out_size', type=int, default=512, help='output dimension (default: 512)')   
parser.add_argument('--in_channels', type=int, default=1, help='input channels (default: 1)')   
parser.add_argument('--out_channels', type=int, default=128, help='output channels (default: 128)')   
parser.add_argument('--epochs', type=int, default=15, help='number of epochs to train feedback weights(default: 15)') 
parser.add_argument('--iter', nargs = '+', type=int, default=[5, 10], help='number of learning iterating of feedback weights layer-wise per batch (default: [5, 10])')
parser.add_argument('--iter_fast', nargs = '+', type=int, default=[5, 10], help='number of learning iterating of feedback weights layer-wise per batch (default: [5, 10])')
parser.add_argument('--batch-size', type=int, default=128, help='batch dimension (default: 128)')   
parser.add_argument('--device-label', type=int, default=0, help='device (default: 1)')   
parser.add_argument('--lr_f', type=float, default=0.05, help='learning rate (default: 0.05)')   
parser.add_argument('--lr_b', nargs = '+', type=float, default=[0.05, 0.05], help='learning rates for the feedback weights (default: [0.05, 0.05])')
parser.add_argument('--lamb', type=float, default=0.01, help='regularization parameter (default: 0.01)')   
parser.add_argument('--beta', type=float, default=0.1, help='nudging parameter (default: 0.1)')   
parser.add_argument('--sym', default=False, action='store_true',help='sets symmetric weight initialization (default: False)')
parser.add_argument('--jacobian', default=False, action='store_true',help='compute jacobians (default: False)')
parser.add_argument('--C', nargs = '+', type=int, default=[32, 64], help='tab of channels (default: [32, 64])')
parser.add_argument('--sigmapi', default=False, action = 'store_true', help='use of sigma-pi G functions (default: False)')
parser.add_argument('--mlp', default=False, action = 'store_true', help='use of MLP G functions (default: False)')
parser.add_argument('--noise', nargs = '+', type=float, default=[0.05, 0.5], help='tab of noise amplitude (default: [0.05, 0.5])')
parser.add_argument('--action', nargs = '+', type=str, default=['test'], help='action and subaction to take (default: [test])')
parser.add_argument('--alg', nargs = '+', type=int, default=[1, 2], help='algorithm used to train the feedback weights layer-wise (default: [1, 2])')
parser.add_argument('--activation', type=str, default='elu', help='activation function in conv layers (default: elu)')
parser.add_argument('--path', type=str, default= None, help='Path directory for the results (default: None)')
parser.add_argument('--last-trial', default=False, action='store_true',help='specifies if the current trial is the last one (default: False)')
parser.add_argument('--warmup', type=int, default=0, help='number of warmup steps (default: 0)')
parser.add_argument('--seed', type=int, default=None, help='seed selected (default: None)')
parser.add_argument('--jac', default=False, action='store_true',help='compute jacobian distance/angle instead of weight distance/angle (default: False)')   
parser.add_argument('--dataset', type=str, default= 'mnist', help='Dataset (default: mnist)')

args = parser.parse_args()  


if args.dataset == 'mnist':
    class ReshapeTransform:
        def __init__(self, new_size):
            self.new_size = new_size

        def __call__(self, img):
            return torch.reshape(img, self.new_size)

    class ReshapeTransformTarget:
        def __init__(self, number_classes):
            self.number_classes = number_classes
        
        def __call__(self, target):
            target=torch.tensor(target).unsqueeze(0).unsqueeze(1)
            target_onehot = torch.zeros((1,self.number_classes))      
            return target_onehot.scatter_(1, target, 1).squeeze(0)


    transforms=[torchvision.transforms.ToTensor()]
    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(root='./data', train=True, download=True,
                             transform=torchvision.transforms.Compose(transforms)
                            ),
    batch_size = args.batch_size, shuffle=True)


    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(root='./data', train=False, download=True,
                             transform=torchvision.transforms.Compose(transforms)
                            ),
    batch_size = args.batch_size, shuffle=True)

elif args.dataset == 'cifar10':
    transform_train = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                                    torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                                                                                           std=(3*0.2023, 3*0.1994, 3*0.2010)) ])   

    transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                                     torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                                                                                      std=(3*0.2023, 3*0.1994, 3*0.2010)) ]) 

    cifar10_train_dset = torchvision.datasets.CIFAR10('./cifar10_pytorch', train=True, transform=transform_train, download=True)
    cifar10_test_dset = torchvision.datasets.CIFAR10('./cifar10_pytorch', train=False, transform=transform_test, download=True)
   
    val_index = np.random.randint(10)
    val_samples = list(range( 5000 * val_index, 5000 * (val_index + 1) ))

    train_loader = torch.utils.data.DataLoader(cifar10_train_dset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(cifar10_test_dset, batch_size=200, shuffle=False, num_workers=1)


if args.device_label >= 0:    
    device = torch.device("cuda:"+str(args.device_label))
else:
    device = torch.device("cpu")


if args.seed is not None:
    seed = args.seed
    torch.manual_seed(seed)
else:
    g = torch.Generator(device = device)
    seed = g.seed()
    torch.manual_seed(seed)

print('Selected seed: {}'.format(seed))


def copy(y, ind_y):
    y_copy = []
    
    for i in range(len(y)):
        y_copy.append(y[i].clone())

    #WATCH OUT: detach previous node!
    y_copy[ind_y - 1] = y_copy[ind_y - 1].detach()    

    return y_copy


class smallNet_benchmark(nn.Module):
    def __init__(self, args):
        super(smallNet_benchmark, self).__init__()
        size = 28
        self.conv1 = nn.Conv2d(1, args.C[0], 5, stride = 2)
        size = np.floor((size - 5)/2 + 1)
        self.conv2 = nn.Conv2d(args.C[0], args.C[1], 5, stride = 2)
        size = int(np.floor((size - 5)/2 + 1))
        self.fc = nn.Linear(args.C[-1]*size**2, 10)
        
        if args.activation == 'elu':
            self.rho = nn.ELU()
        elif args.activation == 'relu':
            self.rho = nn.ReLU()
    
    def forward(self, x):
        out = self.rho(self.conv1(x))
        out = self.rho(self.conv2(out))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class layer_fc(nn.Module):
    def __init__(self, in_size, out_size, alg, iter, noise, last_layer = False):
        super(layer_fc, self).__init__()
        self.f = nn.Linear(in_size, out_size)
        self.b = nn.Linear(out_size, in_size)
        
        self.alg = alg
        self.iter = iter
        self.noise = noise

    def ff(self, x):
        x_flat = x.view(x.size(0), - 1)
        y = self.f(x_flat)
        return y

    def bb(self, x, y):
        r = self.b(y)
        r = r.view(x.size())

        return r        

    def forward(self, x, back = False):
        y = self.ff(x)
    
        if back:
            r = self.bb(x, y)
            return y, r 
        else:
            return y     

    def weight_b_normalize(self, dx, dy, dr):
            
        factor = ((dy**2).sum(-1).mean(0))/((dx*dr).view(dx.size(0), dx.size(1), -1).sum(-1).mean(0)) 
        factor = factor.mean()
      
         
        with torch.no_grad():
            self.b.weight.data = factor*self.b.weight.data

    def weight_b_sym(self):
        with torch.no_grad():
            self.f.weight.data = self.b.weight.data.t()
    
    def compute_dist_angle(self, *args):
        F = self.f.weight
        G = self.b.weight.t()

        dist = torch.sqrt(((F - G)**2).sum()/(F**2).sum())

        F_flat = torch.reshape(F, (F.size(0), -1))
        G_flat = torch.reshape(G, (G.size(0), -1))
        cos_angle = ((F_flat*G_flat).sum(1))/torch.sqrt(((F_flat**2).sum(1))*((G_flat**2).sum(1)))     
        angle = (180.0/np.pi)*(torch.acos(cos_angle).mean().item())

        return dist, angle
    
    def weight_b_train(self, y, optimizer, arg_return = False):
        
        nb_iter = self.iter
        sigma = self.noise

        if self.alg == 1:
            noise_tab = []
            dy_tab = []
            dr_tab = []

            for iter in range(1, nb_iter + 1):
                y_temp, r_temp = self(y, back = True) 
                noise = sigma*torch.randn_like(y)
                y_noise, r_noise = self(y + noise, back = True)
                dy = (y_noise - y_temp)
                dr = (r_noise - r_temp)
                
                loss_b = -(noise*dr).view(dr.size(0), -1).sum(1).mean()

                optimizer.zero_grad() 
                loss_b.backward()            
                optimizer.step()

                noise_tab.append(noise.detach())            
                dy_tab.append(dy.detach())
                dr_tab.append(dr.detach())

            noise_tab = torch.stack(noise_tab, dim=0)
            dy_tab = torch.stack(dy_tab, dim=0)
            dr_tab = torch.stack(dr_tab, dim=0)
            self.weight_b_normalize(noise_tab, dy_tab, dr_tab) 
 
        elif self.alg == 2:
            noise_tab = []
            dy_tab = []
            dr_tab = []

            for iter in range(1, nb_iter + 1):
                y_temp, r_temp = self(y, back = True)
                
                           
                noise = sigma*torch.randn_like(y)
                y_noise, r_noise = self(y + noise, back = True)
                dy = (y_noise - y_temp)
                dr = (r_noise - r_temp)              

                noise_y = sigma*torch.randn_like(y_temp)
                r_noise_y = self.bb(y, y_temp + noise_y)
                dr_y = (r_noise_y - r_temp)
                loss_b = -2*(noise*dr).view(dr.size(0), -1).sum(1).mean() + (dr_y**2).view(dr_y.size(0), -1).sum(1).mean() 
                
                optimizer.zero_grad() 
                loss_b.backward()            
                optimizer.step()

                noise_tab.append(noise)            
                dy_tab.append(dy)
                dr_tab.append(dr)
  
        if arg_return:
            return loss_b
    
    def weight_f_train(self, y, t, optimizer):      
        #update forward weights
        loss_f = 0.5*((y - t)**2).view(y.size(0), -1).sum(1)
        loss_f = loss_f.mean()
        optimizer.zero_grad()
        loss_f.backward(retain_graph = True)

        #DEBUG        
        for name, p in net.named_parameters():
            if p.grad is not None:
                print(name + ' has mean gradient {}'.format(p.grad.mean()))

        optimizer.step()
        
        return loss_f
    
    def propagateError(self, r, t):
        delta = self.bb(r, t) - r
        return delta
 
 
class layer_mlp_fc(nn.Module):
    def __init__(self, in_size, out_size, activation, alg, iter, noise, last_layer = False):
        super(layer_mlp_fc, self).__init__()
        self.f = nn.Linear(in_size, out_size)
        self.last_layer = last_layer
        
        b = nn.ModuleList([nn.Linear(out_size, out_size), nn.Linear(out_size, in_size)])
        self.b = b
        
        if activation == 'elu':
            self.rho = nn.ELU()
        elif activation == 'relu':
            self.rho = nn.ReLU()
        elif activation == 'tanh':
            self.rho == nn.Tanh()
        elif activation == 'sigmoid':
            self.rho == nn.Sigmoid()
        
        self.alg = alg
        self.iter = iter
        self.noise = noise

    def ff(self, x):
        if self.last_layer:
            x_flat = x.view(x.size(0), - 1)
            y = self.f(x_flat)
        else:
            y = self.f(x)
        return y

    def ff_jac(self, x):
        y = self.f(x)
        return y

    def bb(self, x, y):
        r = self.rho(self.b[1](self.rho(self.b[0](y))))

        if self.last_layer:
            r = r.view(x.size())

        return r        
    
    def bb_jac(self, y):
        r = self.rho(self.b[1](self.rho(self.b[0](y))))
        
        return r        

    def forward(self, x, back = False):
        y = self.ff(x)
    
        if back:
            r = self.bb(x, y)
            return y, r 
        else:
            return y     

        
    def compute_dist_angle(self, *args):
        if len(args)> 0:
            x = args[0]
        
        x = x.view(x.size(0), -1)
        F = torch.autograd.functional.jacobian(self.ff_jac, x)
        F = torch.transpose(torch.diagonal(F, dim1=0, dim2=2), 0, 2)
        y = self.ff_jac(x)
        G = torch.autograd.functional.jacobian(self.bb_jac, y) 
        G = torch.transpose(torch.diagonal(G, dim1=0, dim2=2), 0, 2)
        G = torch.transpose(G, 1, 2) 

        dist = torch.sqrt(((F - G)**2).sum(2).sum(1).mean()/(F**2).sum(2).sum(1).mean())

        F_flat = torch.reshape(F, (F.size(0), -1))
        G_flat = torch.reshape(G, (G.size(0), -1))
        cos_angle = ((F_flat*G_flat).sum(1))/torch.sqrt(((F_flat**2).sum(1))*((G_flat**2).sum(1)))     
        angle = (180.0/np.pi)*(torch.acos(cos_angle).mean().item())

        return dist, angle
    
    def weight_b_train(self, y, optimizer, arg_return = False):
        
        nb_iter = self.iter
        sigma = self.noise

        if self.alg == 1:
            raise Exception("Alg 1 does not apply to the class layer_MLP_fc")           
 
        elif self.alg == 2:
            noise_tab = []
            dy_tab = []
            dr_tab = []

            for iter in range(1, nb_iter + 1):
                y_temp, r_temp = self(y, back = True)
                
                           
                noise = sigma*torch.randn_like(y)
                y_noise, r_noise = self(y + noise, back = True)
                dy = (y_noise - y_temp)
                dr = (r_noise - r_temp)              

                noise_y = sigma*torch.randn_like(y_temp)
                r_noise_y = self.bb(y, y_temp + noise_y)
                dr_y = (r_noise_y - r_temp) 
                loss_b = -2*(noise*dr).view(dr.size(0), -1).sum(1).mean() + (dr_y**2).view(dr_y.size(0), -1).sum(1).mean() 
                
                optimizer.zero_grad() 
                loss_b.backward()            
                optimizer.step()

                noise_tab.append(noise)            
                dy_tab.append(dy)
                dr_tab.append(dr)
  
        if arg_return:
            return loss_b

class layer_sigmapi_fc(nn.Module):
    def __init__(self, in_size, out_size, activation, alg, iter, noise, last_layer = False):
        super(layer_sigmapi_fc, self).__init__()
        self.f = nn.Linear(in_size, out_size)
        
        b = nn.ModuleList([nn.Linear(out_size, in_size), nn.Linear(out_size, in_size)])
        self.b = b
        
        if activation == 'elu':
            self.rho = nn.ELU()
        elif activation == 'relu':
            self.rho = nn.ReLU()
        
        self.alg = alg
        self.iter = iter
        self.noise = noise

        self.last_layer = last_layer

    def ff(self, x):
        if self.last_layer:
            x_flat = x.view(x.size(0), - 1)
            y = self.f(x_flat)
        else:
            y = self.f(x)
        return y

    def bb(self, x, y):

        r = self.b[0](y)*self.b[1](y)

        if self.last_layer:
            r = r.view(x.size())
        
        r = self.rho(r)

        return r        


    def forward(self, x, back = False):
        y = self.ff(x)
    
        if back:
            r = self.bb(x, y)
            return y, r 
        else:
            return y     

    def weight_b_normalize(self, dx, dy, dr):
         
        pre_factor = ((dy**2).sum(-1).mean(0))/((dx*dr).view(dx.size(0), dx.size(1), -1).sum(-1).mean(0)) 
        sign_factor = torch.sign(pre_factor)
        factor = torch.sqrt(torch.abs(pre_factor))

        factor = factor.mean()
        sign_factor = torch.sign(sign_factor.mean())
        pos_sign = [1, 1] 
        pos_sign[np.random.randint(2)] = int(sign_factor.item())

        with torch.no_grad():
            self.b[0].weight.data = pos_sign[0]*factor*self.b[0].weight.data
            self.b[1].weight.data = pos_sign[1]*factor*self.b[1].weight.data 

    def compute_dist_angle(self, x):
 
        F = self.f.weight
        y = self.ff(x)
        G = (self.b[0].weight)*(self.b[1](y).mean(0).unsqueeze(1))+ (self.b[1].weight)*(self.b[0](y).mean(0).unsqueeze(1)) 
        G = G.t()

        dist = torch.sqrt(((F - G)**2).sum()/(F**2).sum())
        
        F_flat = torch.reshape(F, (F.size(0), -1))
        G_flat = torch.reshape(G, (G.size(0), -1))
        cos_angle = ((F_flat*G_flat).sum(1))/torch.sqrt(((F_flat**2).sum(1))*((G_flat**2).sum(1)))     
        angle = (180.0/np.pi)*(torch.acos(cos_angle).mean().item())

        return dist, angle

    def weight_b_train(self, y, optimizer, arg_return = False):

        nb_iter = self.iter
        sigma = self.noise        

        if self.alg == 1:
            noise_tab = []
            dy_tab = []
            dr_tab = []

            for iter in range(1, nb_iter + 1):
                y_temp, r_temp = self(y, back = True)
                noise = sigma*torch.randn_like(y)
                y_noise, r_noise = self(y + noise, back = True)
                dy = (y_noise - y_temp)
                dr = (r_noise - r_temp)
                
                loss_b = -(noise*dr).view(dr.size(0), -1).sum(1).mean()
                
                optimizer.zero_grad()
                loss_b.backward()
                optimizer.step()
                
                noise_tab.append(noise)
                dy_tab.append(dy)
                dr_tab.append(dr)
           
            noise_tab = torch.stack(noise_tab, dim=0)
            dy_tab = torch.stack(dy_tab, dim=0)
            dr_tab = torch.stack(dr_tab, dim=0)
            self.weight_b_normalize(noise_tab, dy_tab, dr_tab)
        
        elif self.alg == 2:

            for iter in range(1, nb_iter + 1):
                y_temp, r_temp = self(y, back = True)
                noise = sigma*torch.randn_like(y)
                y_noise, r_noise = self(y + noise, back = True)
                dy = (y_noise - y_temp)
                dr = (r_noise - r_temp)

                noise_y = sigma*torch.randn_like(y_temp)
                r_noise_y = self.bb(y, y_temp + noise_y)
                dr_y = (r_noise_y - r_temp)
                loss_b = -2*(noise*dr).view(dr.size(0), -1).sum(1).mean() + (dr_y**2).view(dr_y.size(0), -1).sum(1).mean() 
                optimizer.zero_grad()
                loss_b.backward()
                optimizer.step()
            
            if arg_return:
                return loss_b

 
class layer_conv(nn.Module):
    def __init__(self, args, in_channels, out_channels, activation, alg=None, iter=None, noise=None):
        super(layer_conv, self).__init__()
        self.f = nn.Conv2d(in_channels, out_channels, 5, stride = 2)
        self.b = nn.ConvTranspose2d(out_channels, in_channels, 5, stride = 2)
       
        if activation == 'elu':
            self.rho = nn.ELU()
        elif activation == 'relu':
            self.rho = nn.ReLU()

        self.alg = alg
        self.iter = iter
        self.noise = noise
        self.jac = args.jac
    
    def ff(self, x):
        y = self.rho(self.f(x))
        return y

    def bb(self, x, y):
        r = self.b(self.rho(y), output_size = x.size())
        return r 

    def forward(self, x, back = False):
        y = self.ff(x)
        
        if back:
            r = self.bb(x, y)
            return y, r
        else:
            return y

    def weight_b_normalize(self, dx, dy, dr):
        
        dy = dy.view(dy.size(0), -1)
        dx = dx.view(dx.size(0), -1)
        dr = dr.view(dr.size(0), -1)
        factor = ((dy**2).sum(1))/((dx*dr).sum(1))
        
        factor = factor.mean()
 
        with torch.no_grad():
            self.b.weight.data = factor*self.b.weight.data    
            
    def weight_b_sym(self):
        with torch.no_grad():
            self.b.weight.data = self.f.weight.data        

    def compute_dist_angle(self, *args):

        if self.jac:
            if len(args)> 0:
                x = args[0]
            
            F = torch.autograd.functional.jacobian(self.ff, x) 
            F = torch.diagonal(F, dim1=0, dim2=4)
            y = self.ff(x)
            G = torch.autograd.functional.jacobian(lambda y: self.bb(x, y), y) 
            G = torch.diagonal(G, dim1=0, dim2=4)
            G = torch.transpose(G, 0, 3)
            G = torch.transpose(G, 1, 4)
            G = torch.transpose(G, 2, 5) 
            
            F = torch.reshape(F, (F.size(-1), -1))
            G = torch.reshape(G, (G.size(-1), -1))
            dist = torch.sqrt(((F - G)**2).sum(1).mean()/(F**2).sum(1).mean())

        else:
            F = self.f.weight
            G = self.b.weight
            dist = torch.sqrt(((F - G)**2).sum()/(F**2).sum())

        
        F_flat = torch.reshape(F, (F.size(0), -1))
        G_flat = torch.reshape(G, (G.size(0), -1))
        cos_angle = ((F_flat*G_flat).sum(1))/torch.sqrt(((F_flat**2).sum(1))*((G_flat**2).sum(1)))     
        angle = (180.0/np.pi)*(torch.acos(cos_angle).mean().item())

        return dist, angle
    
    def weight_b_train(self, y, optimizer, arg_return = False):

        nb_iter = self.iter
        sigma = self.noise

        if self.alg == 1:        
            for iter in range(1, nb_iter + 1):
                y_temp, r_temp = self(y, back = True)
                noise = sigma*torch.randn_like(y)
                y_noise, r_noise = self(y + noise, back = True)
                dy = (y_noise - y_temp)
                dr = (r_noise - r_temp)
                
                loss_b = -(noise*dr).view(dr.size(0), -1).sum(1).mean()

                optimizer.zero_grad()
                loss_b.backward()
                optimizer.step()
               
            self.weight_b_normalize(noise, dy, dr)

        elif self.alg == 2:
            for iter in range(1, nb_iter + 1):
                y_temp, r_temp = self(y, back = True)
                noise = sigma*torch.randn_like(y)
                y_noise, r_noise = self(y + noise, back = True)
                dy = (y_noise - y_temp)
                dr = (r_noise - r_temp)
              
                noise_y = sigma*torch.randn_like(y_temp)
                r_noise_y = self.bb(y, y_temp + noise_y)
                dr_y = (r_noise_y - r_temp)
                loss_b = -2*(noise*dr).view(dr.size(0), -1).sum(1).mean() + (dr_y**2).view(dr_y.size(0), -1).sum(1).mean() 
                
                optimizer.zero_grad()
                loss_b.backward()
                optimizer.step()
     
        if arg_return:
            return loss_b
   
    def weight_f_train(self, y, t, optimizer):      
        #update forward weights
        loss_f = 0.5*((y - t)**2).view(y.size(0), -1).sum(1)
        loss_f = loss_f.mean()
        optimizer.zero_grad()
        loss_f.backward(retain_graph = True)
        
        ''' 
        for name, p in net.named_parameters():
            if p.grad is not None:
                print(name + ' has mean gradient {}'.format(p.grad.mean()))
        '''
        
        optimizer.step()
        
        return loss_f

    def propagateError(self, r, t):
        delta = self.bb(r, t) - r
        return delta

class layer_mlp_conv(nn.Module):
    def __init__(self, in_channels, out_channels, activation, alg=None, iter=None, noise=None):
        super(layer_mlp_conv, self).__init__()
        self.f = nn.Conv2d(in_channels, out_channels, 5, stride = 2)
        self.b = nn.ConvTranspose2d(out_channels, in_channels, 5, stride = 2)

        b = nn.ModuleList([nn.ConvTranspose2d(out_channels, out_channels, 5, stride = 1, padding = 2), nn.ConvTranspose2d(out_channels, in_channels, 5, stride = 2)])
        self.b = b
       
        if activation == 'elu':
            self.rho = nn.ELU()
        elif activation == 'relu':
            self.rho = nn.ReLU()

        self.alg = alg
        self.iter = iter
        self.noise = noise
    
    def ff(self, x):
        y = self.rho(self.f(x))
        return y

    def bb(self, x, y):
        r = self.b[0](self.rho(y), output_size = y.size())
        r = self.b[1](self.rho(r), output_size = x.size())
        return r 

    def forward(self, x, back = False):
        y = self.ff(x)
        
        if back:
            r = self.bb(x, y)
            return y, r
        else:
            return y

    
    def compute_dist_angle(self, *args):
         
        if len(args)> 0:
            x = args[0]
        
        F = torch.autograd.functional.jacobian(self.ff, x) 
        F = torch.diagonal(F, dim1=0, dim2=4)
        y = self.ff(x)
        G = torch.autograd.functional.jacobian(lambda y: self.bb(x, y), y) 
        G = torch.diagonal(G, dim1=0, dim2=4)
        G = torch.transpose(G, 0, 3)
        G = torch.transpose(G, 1, 4)
        G = torch.transpose(G, 2, 5) 
        
        F = torch.reshape(F, (F.size(-1), -1))
        G = torch.reshape(G, (G.size(-1), -1))
        dist = torch.sqrt(((F - G)**2).sum(1).mean()/(F**2).sum(1).mean())
        
        F_flat = torch.reshape(F, (F.size(0), -1))
        G_flat = torch.reshape(G, (G.size(0), -1))
        cos_angle = ((F_flat*G_flat).sum(1))/torch.sqrt(((F_flat**2).sum(1))*((G_flat**2).sum(1)))     
        angle = (180.0/np.pi)*(torch.acos(cos_angle).mean().item())

        return dist, angle
    
    def weight_b_train(self, y, optimizer, arg_return = False):

        nb_iter = self.iter
        sigma = self.noise

        if self.alg == 1:        
            raise Exception("Alg 1 not applicable to class conv_fc_layer")      
 
        elif self.alg == 2:
            for iter in range(1, nb_iter + 1):
                y_temp, r_temp = self(y, back = True)
                noise = sigma*torch.randn_like(y)
                y_noise, r_noise = self(y + noise, back = True)
                dy = (y_noise - y_temp)
                dr = (r_noise - r_temp)
              
                noise_y = sigma*torch.randn_like(y_temp)
                r_noise_y = self.bb(y, y_temp + noise_y)
                dr_y = (r_noise_y - r_temp)
                loss_b = -2*(noise*dr).view(dr.size(0), -1).sum(1).mean() + (dr_y**2).view(dr_y.size(0), -1).sum(1).mean() 
                
                optimizer.zero_grad()
                loss_b.backward()
                optimizer.step()
     
        if arg_return:
            return loss_b

class layer_sigmapi_conv(nn.Module):
    def __init__(self, in_channels, out_channels, activation, alg=None, iter=None, noise=None):
        super(layer_sigmapi_conv, self).__init__()
        self.f = nn.Conv2d(in_channels, out_channels, 5, stride = 2)
        self.b = nn.ConvTranspose2d(out_channels, in_channels, 5, stride = 2)

        b = nn.ModuleList([nn.ConvTranspose2d(out_channels, in_channels, 5, stride = 2) , nn.ConvTranspose2d(out_channels, in_channels, 5, stride = 2)])
        self.b = b
       
        if activation == 'elu':
            self.rho = nn.ELU()
        elif activation == 'relu':
            self.rho = nn.ReLU()

        self.alg = alg
        self.iter = iter
        self.noise = noise
    
    def ff(self, x):
        y = self.rho(self.f(x))
        return y

    def bb(self, x, y):
        r = self.b[0](self.rho(y), output_size = x.size())*self.b[1](self.rho(y), output_size = x.size())
        return r 

    def forward(self, x, back = False):
        y = self.ff(x)
        
        if back:
            r = self.bb(x, y)
            return y, r
        else:
            return y

    
    def compute_dist_angle(self, *args):
         
        if len(args)> 0:
            x = args[0]
        
        F = torch.autograd.functional.jacobian(self.ff, x) 
        F = torch.diagonal(F, dim1=0, dim2=4)
        y = self.ff(x)
        G = torch.autograd.functional.jacobian(lambda y: self.bb(x, y), y) 
        G = torch.diagonal(G, dim1=0, dim2=4)
        G = torch.transpose(G, 0, 3)
        G = torch.transpose(G, 1, 4)
        G = torch.transpose(G, 2, 5) 
        
        F = torch.reshape(F, (F.size(-1), -1))
        G = torch.reshape(G, (G.size(-1), -1))
        dist = torch.sqrt(((F - G)**2).sum(1).mean()/(F**2).sum(1).mean())
        
        F_flat = torch.reshape(F, (F.size(0), -1))
        G_flat = torch.reshape(G, (G.size(0), -1))
        cos_angle = ((F_flat*G_flat).sum(1))/torch.sqrt(((F_flat**2).sum(1))*((G_flat**2).sum(1)))     
        angle = (180.0/np.pi)*(torch.acos(cos_angle).mean().item())

        return dist, angle
    
    def weight_b_train(self, y, optimizer, arg_return = False):

        nb_iter = self.iter
        sigma = self.noise

        if self.alg == 1:        
            raise Exception("Alg 1 not applicable to class conv_fc_layer")      
 
        elif self.alg == 2:
            for iter in range(1, nb_iter + 1):
                y_temp, r_temp = self(y, back = True)
                noise = sigma*torch.randn_like(y)
                y_noise, r_noise = self(y + noise, back = True)
                dy = (y_noise - y_temp)
                dr = (r_noise - r_temp)
              
                noise_y = sigma*torch.randn_like(y_temp)
                r_noise_y = self.bb(y, y_temp + noise_y)
                dr_y = (r_noise_y - r_temp)
                loss_b = -2*(noise*dr).view(dr.size(0), -1).sum(1).mean() + (dr_y**2).view(dr_y.size(0), -1).sum(1).mean() 
                
                optimizer.zero_grad()
                loss_b.backward()
                optimizer.step()
     
        if arg_return:
            return loss_b
    
    def weight_f_train(self, y, t, optimizer):      
        #update forward weights
        loss_f = 0.5*((y - t)**2).view(y.size(0), -1).sum(1)
        loss_f = loss_f.mean()
        optimizer.zero_grad()
        loss_f.backward(retain_graph = True)

        '''       
        for name, p in net.named_parameters():
            if p.grad is not None:
                print(name + ' has mean gradient {}'.format(p.grad.mean()))
        '''
     
        optimizer.step()
         
        return loss_f
    
    def propagateError(self, r, t):
        delta = self.bb(r, t) - r
        return delta
 
class layer_convpool(nn.Module):
    def __init__(self, args, in_channels, out_channels, activation, iter=None, noise=None):
        super(layer_convpool, self).__init__()
        self.f = nn.Conv2d(in_channels, out_channels, 3, stride = 1, padding = 1)
        self.pool = nn.MaxPool2d(2, stride = 2, return_indices = True)            
        
        if iter is not None:
            self.b = nn.ConvTranspose2d(out_channels, in_channels, 3, stride = 1, padding = 1)
            self.unpool = nn.MaxUnpool2d(2, stride = 2)  
      
        #******************************# 
        if activation == 'elu':
            self.rho = nn.ELU()
        elif activation == 'relu':
            self.rho = nn.ReLU()
        #******************************#

        #******************#
        self.iter = iter
        self.noise = noise
        self.jac = args.jac
        #******************#
    
    def ff(self, x, ret_ind = False):
        y, ind = self.pool(self.rho(self.f(x)))
        
        if ret_ind:
            return y, ind
        else:
            return y

    def bb(self, x, y, ind):
        r = self.unpool(y, ind, output_size = x.size())
        r = self.b(self.rho(r), output_size = x.size())
        
        return r 

    def forward(self, x, back = False):
        
        if back:
            y, ind = self.ff(x, ret_ind = True)
            r = self.bb(x, y, ind)
            return y, (r, ind)
        else:
            y = self.ff(x)
            return y
       
    def weight_b_sym(self):
        with torch.no_grad():
            self.b.weight.data = self.f.weight.data        

    def compute_dist_angle(self, *args):

        if self.jac:
            if len(args)> 0:
                x = args[0]
            
            F = torch.autograd.functional.jacobian(self.ff, x) 
            F = torch.diagonal(F, dim1=0, dim2=4)
            y, ind = self.ff(x, ret_ind = True)
            G = torch.autograd.functional.jacobian(lambda y: self.bb(x, y, ind), y) 
            G = torch.diagonal(G, dim1=0, dim2=4)
            G = torch.transpose(G, 0, 3)
            G = torch.transpose(G, 1, 4)
            G = torch.transpose(G, 2, 5) 
            
            F = torch.reshape(F, (F.size(-1), -1))
            G = torch.reshape(G, (G.size(-1), -1))
            dist = torch.sqrt(((F - G)**2).sum(1).mean()/(F**2).sum(1).mean())

        else:
            F = self.f.weight
            G = self.b.weight
            dist = torch.sqrt(((F - G)**2).sum()/(F**2).sum())

        
        F_flat = torch.reshape(F, (F.size(0), -1))
        G_flat = torch.reshape(G, (G.size(0), -1))
        cos_angle = ((F_flat*G_flat).sum(1))/torch.sqrt(((F_flat**2).sum(1))*((G_flat**2).sum(1)))     
     
        if arg_return:
            return loss_b

    def weight_b_train(self, y, optimizer, arg_return = False):
        
        nb_iter = self.iter
        sigma = self.noise
            
        for iter in range(1, nb_iter + 1):
            y_temp, r_temp, ind = self(y, back = True)
            noise = sigma*torch.randn_like(y)
            y_noise, r_noise, ind_noise = self(y + noise, back = True)
            dy = (y_noise - y_temp)
            dr = (r_noise - r_temp)
          
            noise_y = sigma*torch.randn_like(y_temp)
            r_noise_y = self.bb(y, y_temp + noise_y, ind)
            dr_y = (r_noise_y - r_temp)
            loss_b = -2*(noise*dr).view(dr.size(0), -1).sum(1).mean() + (dr_y**2).view(dr_y.size(0), -1).sum(1).mean() 
            
            optimizer.zero_grad()
            loss_b.backward()
            optimizer.step()
     
        if arg_return:
            return loss_b

    def weight_f_train(self, y, t, optimizer):
        loss_f = 0.5*((y - t)**2).view(y.size(0), -1).sum(1)
        loss_f = loss_f.mean()
        optimizer.zero_grad()
        loss_f.backward(retain_graph = True)
        optimizer.step()
        
        #DEBUG
        for name, p in net.named_parameters():
            if p.grad is not None:
                print(name + ' has mean gradient {}'.format(p.grad.mean()))
        
        
        return loss_f

    def propagateError(self, r_tab, t):
        r = r_tab[0]
        ind = r_tab[1]
        delta = self.bb(r_tab[0], t, r_tab[1]) - r_tab[0]

        return delta

class smallNet(nn.Module):
    def __init__(self, args):
        super(smallNet, self).__init__()

        #MNIST        
        size = 28
        args.C = [1] + args.C

        layers = nn.ModuleList([])

        layers.append(layer_conv(args, args.C[0], args.C[1], args.activation))
        size = int(np.floor((size - 5)/2 + 1))

        if args.mlp:
            for i in range(len(args.C) - 2):
                layers.append(layer_mlp_conv(args.C[i + 1], args.C[i + 2], 
                                        args.activation, args.alg[i], 
                                        args.iter[i], args.noise[i]))
                
                size = int(np.floor((size - 5)/2 + 1))
        elif args.sigmapi:
            for i in range(len(args.C) - 2):
                layers.append(layer_sigmapi_conv(args.C[i + 1], args.C[i + 2], 
                                        args.activation, args.alg[i], 
                                        args.iter[i], args.noise[i]))
                
                size = int(np.floor((size - 5)/2 + 1))
        else:
            for i in range(len(args.C) - 2):
                layers.append(layer_conv(args, args.C[i + 1], args.C[i + 2], 
                                        args.activation, args.alg[i], 
                                        args.iter[i], args.noise[i]))
                
                size = int(np.floor((size - 5)/2 + 1))
 
        if args.sigmapi:
            #***************WATCH OUT: changed for training!*********************#
            '''
            layers.append(layer_sigmapi_fc((size**2)*args.C[-1], 10, 
                                    args.activation, args.alg[-1], args.iter[-1],
                                    args.noise[-1], last_layer = True))
            '''

            
            layers.append(layer_fc((size**2)*args.C[-1], 10, 
                                    args.alg[-1], args.iter[-1],
                                    args.noise[-1], last_layer = True))
            
            #********************************************************************#
        elif args.mlp:        
            layers.append(layer_mlp_fc((size**2)*args.C[-1], 10, 
                                    args.activation, args.alg[-1], args.iter[-1],
                                    args.noise[-1], last_layer = True))
        else:
            layers.append(layer_fc((size**2)*args.C[-1], 10, 
                                    args.alg[-1], args.iter[-1],
                                    args.noise[-1], last_layer = True))


        self.layers = layers
        self.logsoft = nn.LogSoftmax(dim=1) 
        self.noise = args.noise
        self.beta = args.beta

    def forward(self, x, ind = None):
        s = x

        if ind is None:
            for i in range(len(self.layers)):
                s  = self.layers[i](s)
       
            return s

        else:
            for i in range(ind):
                if i == ind - 1:
                    s = s.detach()
                    s.requires_grad = True
                    s, r = self.layers[i](s, back = True)
                else:
                    s = self.layers[i](s) 

            return s, r

    def weight_b_sym(self):
        for i in range(len(self.layers)):
            self.layers[i].weight_b_sym()

    def weight_f_train(self, y, r, t, id_layer, optimizer, beta):      
        #update forward weights
        loss_f = 0.5*((y - t)**2).view(y.size(0), -1).sum(1)
        loss_f = loss_f.mean()
        optimizer.zero_grad()
        loss_f.backward(retain_graph = True)
        
        optimizer.step()
         
        #compute previous targets         
        if (id_layer < len(self.layers) - 1):
            delta = self.layers[-1 - id_layer].bb(r, t) - r
            y, r = self(data, ind = len(self.layers) - 1 - id_layer)
            t = (y + delta).detach()

        return y, r, t, loss_f

def test(net, test_loader):

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

    test_acc = 100.*correct/total

    return test_acc


class ResNet(nn.Module):
    def __init__(self, args):
        super(ResNet, self).__init__()

        #CIFAR-10       
        size = 32
        args.C = [3] + args.C

        layers = nn.ModuleList([])

        layers.append(layer_convpool(args, args.C[0], args.C[1], args.activation))
        size = int(np.floor(size/2))

        for i in range(len(args.C) - 2):
            layers.append(layer_convpool(args, args.C[i + 1], args.C[i + 2], 
                                    args.activation, args.iter[i], args.noise[i]))
            
            size = int(np.floor(size/2))
 
        layers.append(layer_fc((size**2)*args.C[-1], 10, 
                                args.alg[-1], args.iter[-1],
                                args.noise[-1], last_layer = True))

        self.layers = layers
        self.logsoft = nn.LogSoftmax(dim=1) 
        self.noise = args.noise
        self.beta = args.beta

    def forward(self, x, ind_layer = None):
        s = x

        if ind_layer is None:
            for i in range(len(self.layers)):
                s  = self.layers[i](s)
       
            return s

        else:
            for i in range(ind_layer):
                if i == ind_layer - 1:
                    s = s.detach()
                    s.requires_grad = True
                    s, r = self.layers[i](s, back = True)
                else:
                    s = self.layers[i](s) 

            return s, r

    def weight_b_sym(self):
        for i in range(len(self.layers)):
            self.layers[i].weight_b_sym()


if __name__ == '__main__':

    if args.path is not None:    
        BASE_PATH = createPath(args)
        command_line = ' '.join(sys.argv) 
        createHyperparameterfile(BASE_PATH, command_line, seed, args)


    if args.dataset == 'mnist':
        net = smallNet(args)

    elif args.dataset == 'cifar10':
        net = ResNet(args)
    
    net = net.to(device)
    print(net)    

    if args.action[0] == 'train': 
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        optim_params_f = []
        optim_params_b = []

        for i in range(len(net.layers)):
            optim_params_f.append({'params': net.layers[i].f.parameters(), 'lr': args.lr_f})
            
        for i in range(len(net.layers) - 1):
            optim_params_b.append({'params': net.layers[i + 1].b.parameters(), 'lr': args.lr_b[i]})

        optimizer_f = torch.optim.SGD(optim_params_f, momentum = 0.9) 
        optimizer_b = torch.optim.SGD(optim_params_b, momentum = 0.9)

        net.train()
                
        train_acc = []
        test_acc = []

        #**STANDARD GENERAL TRAINING WITH FORWARD AND FEEDBACK WEIGHTS**#
        for epoch in range(args.epochs): 
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (data, target) in enumerate(train_loader):

                data, target = data.to(device), target.to(device)
                
                #****FEEDBACK WEIGHTS****#          
                y = net.layers[0](data).detach() 
                for id_layer in range(len(net.layers) - 1):                     
                    net.layers[id_layer + 1].weight_b_train(y, optimizer_b)
                    y = net.layers[id_layer + 1](y).detach()

                pred = torch.exp(net.logsoft(y)) 

                #*********FORWARD WEIGHTS********#
                y, r = net(data, ind = len(net.layers))
                
                L = criterion(y.float(), target).squeeze()
                init_grads = torch.tensor([1 for i in range(y.size(0))], dtype=torch.float, device=device, requires_grad=True) 
                grads = torch.autograd.grad(L, y, grad_outputs=init_grads, create_graph = True)
                delta = -args.beta*grads[0]
                
                t = y + delta

                for id_layer in range(len(net.layers)):        
                    
                    loss_f = net.layers[-1 - id_layer].weight_f_train(y, t, optimizer_f)
      
                    #compute previous targets         
                    if (id_layer < len(net.layers) - 1):
                        delta = net.layers[-1 - id_layer].propagateError(r, t)
                        y, r = net(data, ind = len(net.layers) - 1 - id_layer)
                        t = (y + delta).detach()
                    
                    if id_layer == 0:
                        loss = loss_f

                train_loss += loss.item()
                _, predicted = pred.max(1)
                targets = target
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Train Acc: %.3f%% (%d/%d)'% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
              
            train_acc.append(100.*correct/total)
            test_acc_temp = test(net, test_loader)
            test_acc.append(test_acc_temp)

            if args.path is not None:
                results = {'train_acc' : train_acc, 'test_acc' : test_acc}
                outfile = open(os.path.join(BASE_PATH, 'results'), 'wb')
                pickle.dump(results, outfile)
                outfile.close()

            if train_acc[-1] < 80: exit()                

    elif args.action[0] == 'test':
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
           
        optim_params_b = []
            
        for i in range(len(net.layers) - 1):
            optim_params_b.append({'params': net.layers[i + 1].b.parameters(), 'lr': args.lr_b[i]})

        optimizer_b = torch.optim.SGD(optim_params_b, momentum = 0.9)
        
        weight_tab = []
        loss_tab = []

        weight_tab = {'dist_weight' : [], 'angle_weight' : []}        

        for batch_iter in range(args.epochs):
            _, (data, target) = next(enumerate(train_loader))             
            data, target = data.to(device), target.to(device)
                                       
            y = net.layers[0](data).detach()
            print('\n Batch iteration {}'.format(batch_iter + 1))
            
            for id_layer in range(len(net.layers) - 1):  
                if (len(args.action) == 1) or id_layer == int(args.action[1]):
                    loss_b = net.layers[id_layer + 1].weight_b_train(y, optimizer_b)
    
                    if (batch_iter + 1) % 10  == 0: 
                        dist_weight, angle_weight = net.layers[id_layer + 1].compute_dist_angle(y)
                        weight_tab['dist_weight'].append(dist_weight)
                        weight_tab['angle_weight'].append(angle_weight)
                        
                        if id_layer < len(net.layers) - 2:
                            layer_str = 'Conv layer ' + str(id_layer + 1) + ': '
                        else:
                            layer_str = 'FC layer:'  

                        print(layer_str)
                        print('Distance between weights: {:.2f}'.format(dist_weight))
                        print('Weight angle: {:.2f} deg'.format(angle_weight))         

                #go to the next layer
                y = net.layers[id_layer + 1](y).detach()

            if args.path is not None:
                results = {'weight_tab' : weight_tab}
                outfile = open(os.path.join(BASE_PATH, 'results'), 'wb')
                pickle.dump(results, outfile)
                outfile.close() 

        
    elif args.action[0] == 'debug':

        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        optim_params_f = []
        optim_params_b = []
        for i in range(len(net.layers)):
            optim_params_f.append({'params': net.layers[i].f.parameters(), 'lr': args.lr_f})
            
        for i in range(len(net.layers) - 1):
            optim_params_b.append({'params': net.layers[i + 1].b.parameters(), 'lr': args.lr_b[i]})
        optimizer_f = torch.optim.SGD(optim_params_f, momentum = 0.9) 
        optimizer_b = torch.optim.SGD(optim_params_b, momentum = 0.9)

        _, (data, target) = next(enumerate(train_loader))
        data, target = data.to(device), target.to(device)

        #Testing forward pass
        '''
        y = net(data)
        print('Forward pass works!')        
        '''
        
        #Testing feedback weights training
        '''        
        y = net.layers[0](data).detach() 
        for id_layer in range(len(net.layers) - 1):                     
            if id_layer < len(net.layers) - 2:
                print('Training feedback convpool layer ' + str(id_layer + 1))
            else:
                print('Training feedback FC layer')  

            net.layers[id_layer + 1].weight_b_train(y, optimizer_b)
            
            #go to the next layer
            y = net.layers[id_layer + 1](y).detach()

        optimizer_b.zero_grad()
        print('Feedback weights training works!')        
        '''

        #Testing forward weights training
        
        y, r = net(data, ind_layer = len(net.layers))
        
        L = criterion(y.float(), target).squeeze()
        init_grads = torch.tensor([1 for i in range(y.size(0))], dtype=torch.float, device=device, requires_grad=True) 
        grads = torch.autograd.grad(L, y, grad_outputs=init_grads, create_graph = True)
        delta = -args.beta*grads[0]
        
        t = y + delta

        for id_layer in range(len(net.layers)):        
            print('Step {}'.format(id_layer + 1))
            loss_f = net.layers[-1 - id_layer].weight_f_train(y, t, optimizer_f)
    
            #compute previous targets         
            if (id_layer < len(net.layers) - 1):
                delta = net.layers[-1 - id_layer].propagateError(r, t)
                y, r = net(data, ind_layer = len(net.layers) - 1 - id_layer)
                t = (y + delta).detach()

            if id_layer == 0:
                loss = loss_f
        
        print('Forward weights training works!')       

