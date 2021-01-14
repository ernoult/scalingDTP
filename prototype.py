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
parser.add_argument('--iter', type=int, default=20, help='number of iterationson feedback weights per batch samples (default: 20)') 
parser.add_argument('--batch-size', type=int, default=128, help='batch dimension (default: 128)')   
parser.add_argument('--device-label', type=int, default=0, help='device (default: 1)')   
parser.add_argument('--lr_f', type=float, default=0.05, help='learning rate (default: 0.05)')   
parser.add_argument('--lr_b', type=float, default=0.5, help='learning rate of the feedback weights (default: 0.5)')   
parser.add_argument('--lamb', type=float, default=0.01, help='regularization parameter (default: 0.01)')   
parser.add_argument('--beta', type=float, default=0.1, help='nudging parameter (default: 0.1)')   
parser.add_argument('--seed', default=False, action='store_true',help='fixes the seed to 1 (default: False)')
parser.add_argument('--sym', default=False, action='store_true',help='sets symmetric weight initialization (default: False)')
parser.add_argument('--jacobian', default=False, action='store_true',help='compute jacobians (default: False)')
parser.add_argument('--C', nargs = '+', type=int, default=[32, 64], help='tab of channels (default: [32, 64])')
parser.add_argument('--sigmapi', default=False, action = 'store_true', help='use of sigma-pi G functions (default: False)')
parser.add_argument('--noise', nargs = '+', type=float, default=[0.05, 0.5], help='tab of noise amplitude (default: [0.05, 0.5])')
parser.add_argument('--action', type=str, default='train', help='action to execute (default: train)')
parser.add_argument('--alg', type=int, default=1, help='algorithm used for feedback weights training (default: 1)') 
parser.add_argument('--activation', type=str, default='elu', help='activation function in conv layers (default: elu)')
parser.add_argument('--path', type=str, default= None, help='Path directory for the results (default: None)')
parser.add_argument('--last-trial', default=False, action='store_true',help='specifies if the current trial is the last one (default: False)')

args = parser.parse_args()  

if args.seed:
    torch.manual_seed(1)


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


#if (args.conv):
transforms=[torchvision.transforms.ToTensor()]
#else:
#transforms=[torchvision.transforms.ToTensor(),ReshapeTransform((-1,))]


train_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST(root='./data', train=True, download=True,
                         transform=torchvision.transforms.Compose(transforms)
                        #,target_transform=ReshapeTransformTarget(10)
                        ),
batch_size = args.batch_size, shuffle=True)


test_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST(root='./data', train=False, download=True,
                         transform=torchvision.transforms.Compose(transforms)
                        #,target_transform=ReshapeTransformTarget(10)
                        ),
batch_size = args.batch_size, shuffle=True)


if args.device_label >= 0:    
    device = torch.device("cuda:"+str(args.device_label))
else:
    device = torch.device("cpu")


def copy(y, ind_y):
    y_copy = []
    
    for i in range(len(y)):
        y_copy.append(y[i].clone())

    #WATCH OUT: detach previous node!
    y_copy[ind_y - 1] = y_copy[ind_y - 1].detach()    

    return y_copy

def compute_jacobians(net, x, y):
    jac_F = torch.autograd.functional.jacobian(net.ff, x) 
    jac_F = torch.transpose(torch.diagonal(jac_F, dim1=0, dim2=2), 0, 2)
    jac_G = torch.autograd.functional.jacobian(net.bb, y) 
    jac_G = torch.transpose(torch.diagonal(jac_G, dim1=0, dim2=2), 0, 2)
   
    jac_G = torch.transpose(jac_G, 1, 2) 
 

    return jac_F, jac_G

class smallNet_benchmark(nn.Module):
    def __init__(self):
        super(smallNet_benchmark, self).__init__()
        size = 28
        self.conv1 = nn.Conv2d(1, 128, 5, stride = 2)
        size = np.floor((size - 5)/2 + 1)
        self.conv2 = nn.Conv2d(128, 256, 5, stride = 2)
        size = int(np.floor((size - 5)/2 + 1))
        self.fc = nn.Linear(256*size**2, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class layer_fc(nn.Module):
    def __init__(self, in_size, out_size, args, last_layer = False):
        super(layer_fc, self).__init__()
        self.f = nn.Linear(in_size, out_size)
        self.b = nn.Linear(out_size, in_size)
        self.last_layer = last_layer
        self.alg = args.alg

    def ff(self, x):
        if self.last_layer:
            x_flat = x.view(x.size(0), - 1)
            y = self.f(x_flat)
        else:
            y = self.f(x)
        return y

    def bb(self, x, y):
        r = self.b(y)
        
        if self.last_layer:
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
      
       
        ''' 
        num = (1/(3*(args.noise[-1])**2))*(((dy**2).sum(-1, keepdim = True))*((dx**2).view(dx.size(0),dx.size(1), -1))).mean(0)
        denom = (dx*dr).view(dx.size(0), dx.size(1), -1).mean(0)
        factor = (num/denom)
        
        factor = torch.where(denom == 0, torch.ones_like(factor), factor)
        factor = factor.mean(0).unsqueeze(1)
        '''
         
        with torch.no_grad():
            self.b.weight.data = factor*self.b.weight.data

    def weight_b_sym(self):
        with torch.no_grad():
            self.f.weight.data = self.b.weight.data.t()
    
    def compute_dist_angle(self, *args):
        F = self.f.weight
        G = self.b.weight.t()

        #************************************************#
         #dist = ((F - G)**2).sum()
        dist = torch.sqrt(((F - G)**2).sum()/(F**2).sum())
        #************************************************#

        F_flat = torch.reshape(F, (F.size(0), -1))
        G_flat = torch.reshape(G, (G.size(0), -1))
        cos_angle = ((F_flat*G_flat).sum(1))/torch.sqrt(((F_flat**2).sum(1))*((G_flat**2).sum(1)))     
        angle = (180.0/np.pi)*(torch.acos(cos_angle).mean().item())

        return dist, angle
    
    def weight_b_train(self, y, nb_iter, optimizer, sigma, arg_return = False):
        
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
                
                #*****************************************************#        
                loss_b = -(noise*dr).view(dr.size(0), -1).sum(1).mean()
                #*****************************************************#

                optimizer.zero_grad() 
                loss_b.backward()            
                optimizer.step()

                noise_tab.append(noise.detach())            
                dy_tab.append(dy.detach())
                dr_tab.append(dr.detach())

            #*************************************************# 
            noise_tab = torch.stack(noise_tab, dim=0)
            dy_tab = torch.stack(dy_tab, dim=0)
            dr_tab = torch.stack(dr_tab, dim=0)
            self.weight_b_normalize(noise_tab, dy_tab, dr_tab) 
            #*************************************************#
 
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

                #***************************************#
                noise_y = sigma*torch.randn_like(y_temp)
                r_noise_y = self.bb(y, y_temp + noise_y)
                dr_y = (r_noise_y - r_temp) 
                loss_b = -2*(noise*dr).view(dr.size(0), -1).sum(1).mean() + (dr_y**2).view(dr_y.size(0), -1).sum(1).mean() 
                #***************************************#
                
                optimizer.zero_grad() 
                loss_b.backward()            
                optimizer.step()

                noise_tab.append(noise)            
                dy_tab.append(dy)
                dr_tab.append(dr)
  
        if arg_return:
            return loss_b

class layer_sigmapi_fc(nn.Module):
    def __init__(self, in_size, out_size, args, last_layer = False):
        super(layer_sigmapi_fc, self).__init__()
        self.f = nn.Linear(in_size, out_size)
        
        #*****************************************************************************#
        b = nn.ModuleList([nn.Linear(out_size, in_size), nn.Linear(out_size, in_size)])
        self.b = b
        #*****************************************************************************#
        
        #******************************# 
        if args.activation == 'elu':
            self.rho = nn.ELU()
        elif args.activation == 'relu':
            self.rho = nn.ReLU()
        #******************************#

        self.alg = args.alg
        self.last_layer = last_layer

    def ff(self, x):
        if self.last_layer:
            x_flat = x.view(x.size(0), - 1)
            y = self.f(x_flat)
        else:
            y = self.f(x)
        return y

    def bb(self, x, y):

        #***************************#
        r = self.b[0](y)*self.b[1](y)
        #***************************#

        if self.last_layer:
            r = r.view(x.size())
        
        #*************#
        r = self.rho(r)
        #*************#

        return r        


    def forward(self, x, back = False):
        y = self.ff(x)
    
        if back:
            r = self.bb(x, y)
            return y, r 
        else:
            return y     

    def weight_b_normalize(self, dx, dy, dr):
         
        #************************************************************************************************#
        pre_factor = ((dy**2).sum(-1).mean(0))/((dx*dr).view(dx.size(0), dx.size(1), -1).sum(-1).mean(0)) 
        sign_factor = torch.sign(pre_factor)
        factor = torch.sqrt(torch.abs(pre_factor))
        #************************************************************************************************#

        factor = factor.mean()
        sign_factor = torch.sign(sign_factor.mean())
        pos_sign = [1, 1] 
        pos_sign[np.random.randint(2)] = int(sign_factor.item())

        with torch.no_grad():
            #***************************************************************#
            self.b[0].weight.data = pos_sign[0]*factor*self.b[0].weight.data
            self.b[1].weight.data = pos_sign[1]*factor*self.b[1].weight.data 
            #***************************************************************#

    def compute_dist_angle(self, x):
 
        #********************# 
        F = self.f.weight
        y = self.ff(x)
        G = (self.b[0].weight)*(self.b[1](y).mean(0).unsqueeze(1))+ (self.b[1].weight)*(self.b[0](y).mean(0).unsqueeze(1)) 
        G = G.t()
        #********************#

        #************************************************#
        #dist = ((F - G)**2).sum()
        dist = torch.sqrt(((F - G)**2).sum()/(F**2).sum())
        #************************************************#
        
        F_flat = torch.reshape(F, (F.size(0), -1))
        G_flat = torch.reshape(G, (G.size(0), -1))
        cos_angle = ((F_flat*G_flat).sum(1))/torch.sqrt(((F_flat**2).sum(1))*((G_flat**2).sum(1)))     
        angle = (180.0/np.pi)*(torch.acos(cos_angle).mean().item())

        return dist, angle

    def weight_b_train(self, y, nb_iter, optimizer, sigma, arg_return = False):
        
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
                
                #*****************************************************#
                loss_b = -(noise*dr).view(dr.size(0), -1).sum(1).mean()
                #*****************************************************#
                
                optimizer.zero_grad()
                loss_b.backward()
                optimizer.step()
                
                noise_tab.append(noise)
                dy_tab.append(dy)
                dr_tab.append(dr)
           
            #************************************************# 
            noise_tab = torch.stack(noise_tab, dim=0)
            dy_tab = torch.stack(dy_tab, dim=0)
            dr_tab = torch.stack(dr_tab, dim=0)
            self.weight_b_normalize(noise_tab, dy_tab, dr_tab)
            #************************************************# 
        
        elif self.alg == 2:

            for iter in range(1, nb_iter + 1):
                y_temp, r_temp = self(y, back = True)
                noise = sigma*torch.randn_like(y)
                y_noise, r_noise = self(y + noise, back = True)
                dy = (y_noise - y_temp)
                dr = (r_noise - r_temp)

                #**************************************#                
                noise_y = sigma*torch.randn_like(y_temp)
                r_noise_y = self.bb(y, y_temp + noise_y)
                dr_y = (r_noise_y - r_temp)
                loss_b = -2*(noise*dr).view(dr.size(0), -1).sum(1).mean() + (dr_y**2).view(dr_y.size(0), -1).sum(1).mean() 
                #**************************************#
                optimizer.zero_grad()
                loss_b.backward()
                optimizer.step()
            
            if arg_return:
                return loss_b

 
class layer_conv(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super(layer_conv, self).__init__()
        self.f = nn.Conv2d(in_channels, out_channels, 5, stride = 2)
        self.b = nn.ConvTranspose2d(out_channels, in_channels, 5, stride = 2)
        self.alg = args.alg 
       
        #******************************# 
        if args.activation == 'elu':
            self.rho = nn.ELU()
        elif args.activation == 'relu':
            self.rho = nn.ReLU()
        #******************************#

    def ff(self, x):
        y = self.rho(self.f(x))
        #y = F.relu(self.f(x))
        #y = self.f(x)
        return y

    def bb(self, x, y):
        r = self.rho(self.b(y, output_size = x.size()))
        #r = F.relu(self.b(y, output_size = x.size()))
        #r = self.b(y, output_size = x.size())
        return r 

    def forward(self, x, back = False):
        y = self.ff(x)
        
        if back:
            r = self.bb(x, y)
            return y, r
        else:
            return y

    def weight_b_normalize(self, dx, dy, dr):
        
        #first technique: same normalization for all out fmaps                
        dy = dy.view(dy.size(0), -1)
        dx = dx.view(dx.size(0), -1)
        dr = dr.view(dr.size(0), -1)
        factor = ((dy**2).sum(1))/((dx*dr).sum(1))
        
        #dy = dy.view(dy.size(0), dy.size(1), -1)
        #dx = dx.view(dx.size(0), dx.size(1), -1)
        #dr = dr.view(dr.size(0), dr.size(1), -1)
        #factor = ((dy**2).sum(-1).mean(0))/((dx*dr).view(dx.size(0), dx.size(1), -1).sum(-1).mean(0)) 

        factor = factor.mean()
 
        with torch.no_grad():
            self.b.weight.data = factor*self.b.weight.data    

        #second technique: fmaps-wise normalization
        '''
        dy_square = ((dy.view(dy.size(0), dy.size(1), -1))**2).sum(-1) 
        dx = dx.view(dx.size(0), dx.size(1), -1)
        dr = dr.view(dr.size(0), dr.size(1), -1)
        dxdr = (dx*dr).sum(-1)
        
        factor = torch.bmm(dy_square.unsqueeze(-1), dxdr.unsqueeze(-1).transpose(1,2)).mean(0)
       
        factor = factor.view(factor.size(0), factor.size(1), 1, 1)
         
        with torch.no_grad():
            self.b.weight.data = factor*self.b.weight.data
        '''
    
    def weight_b_sym(self):
        with torch.no_grad():
            self.b.weight.data = self.f.weight.data        

    def compute_dist_angle(self, *args):
        F = self.f.weight
        G = self.b.weight
        
        #************************************************#
        #dist = ((F - G)**2).sum()
        dist = torch.sqrt(((F - G)**2).sum()/(F**2).sum())
        #************************************************#

        F_flat = torch.reshape(F, (F.size(0), -1))
        G_flat = torch.reshape(G, (G.size(0), -1))
        cos_angle = ((F_flat*G_flat).sum(1))/torch.sqrt(((F_flat**2).sum(1))*((G_flat**2).sum(1)))     
        angle = (180.0/np.pi)*(torch.acos(cos_angle).mean().item())

        return dist, angle
    
    def weight_b_train(self, y, nb_iter, optimizer, sigma, arg_return = False):

        if self.alg == 1:        
            #noise_tab = []
            #dy_tab = []
            #dr_tab = []

            for iter in range(1, nb_iter + 1):
                y_temp, r_temp = self(y, back = True)
                noise = sigma*torch.randn_like(y)
                y_noise, r_noise = self(y + noise, back = True)
                dy = (y_noise - y_temp)
                dr = (r_noise - r_temp)
                
                #*****************************************************#
                loss_b = -(noise*dr).view(dr.size(0), -1).sum(1).mean()
                #*****************************************************#

                optimizer.zero_grad()
                loss_b.backward()
                optimizer.step()
               
                #noise_tab.append(noise.detach())
                #dy_tab.append(dy.detach())
                #dr_tab.append(dr.detach())        

            #renormalize once per sample
            self.weight_b_normalize(noise, dy, dr)
            #noise_tab = torch.stack(noise_tab, dim=0)
            #dy_tab = torch.stack(dy_tab, dim=0)
            #dr_tab = torch.stack(dr_tab, dim=0)
            #self.weight_b_normalize(noise_tab, dy_tab, dr_tab)

        elif self.alg == 2:
            for iter in range(1, nb_iter + 1):
                y_temp, r_temp = self(y, back = True)
                noise = sigma*torch.randn_like(y)
                y_noise, r_noise = self(y + noise, back = True)
                dy = (y_noise - y_temp)
                dr = (r_noise - r_temp)
              
                #**************************************# 
                noise_y = sigma*torch.randn_like(y_temp)
                r_noise_y = self.bb(y, y_temp + noise_y)
                dr_y = (r_noise_y - r_temp)
                loss_b = -2*(noise*dr).view(dr.size(0), -1).sum(1).mean() + (dr_y**2).view(dr_y.size(0), -1).sum(1).mean() 
                #**************************************#
                
                optimizer.zero_grad()
                loss_b.backward()
                optimizer.step()
     
        if arg_return:
            return loss_b

class layer_sigmapi_conv(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super(layer_sigmapi_conv, self).__init__()
        self.f = nn.Conv2d(in_channels, out_channels, 5, stride = 2)
        
        #*****************************************************************************#
        b = nn.ModuleList([nn.ConvTranspose2d(out_channels, in_channels, 5, stride = 2), nn.ConvTranspose2d(out_channels, in_channels, 5, stride = 2)])
        self.b = b
        #*****************************************************************************#
        
        self.alg = args.alg 


    def ff(self, x):
        #y = F.relu(self.f(x))
        y = self.f(x)
        return y

    def bb(self, x, y):
        
        #***************************************************************************#
        r = self.b[0](y, output_size = x.size())*self.b[0](y, output_size = x.size())
        #***************************************************************************#

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
 
        #dy = dy.view(dy.size(0), dy.size(1), -1)
        #dx = dx.view(dx.size(0), dx.size(1), -1)
        #dr = dr.view(dr.size(0), dr.size(1), -1) 
        #pre_factor = ((dy**2).sum(-1).mean(0))/((dx*dr).view(dx.size(0), dx.size(1), -1).sum(-1).mean(0)) 
        
        #*****************************************#
        sign_factor = torch.sign(pre_factor)
        factor = torch.sqrt(torch.abs(pre_factor))
        #*****************************************#

        factor = factor.mean()
        sign_factor = torch.sign(sign_factor.mean())
        pos_sign = [1, 1] 
        pos_sign[np.random.randint(2)] = int(sign_factor.item())

        with torch.no_grad():
            #***************************************************************#
            self.b[0].weight.data = pos_sign[0]*factor*self.b[0].weight.data
            self.b[1].weight.data = pos_sign[1]*factor*self.b[1].weight.data 
            #***************************************************************#

    
    def weight_b_sym(self):
        with torch.no_grad():
            self.b.weight.data = self.f.weight.data        

    def compute_dist_angle(self, *args):
        F = self.f.weight
        G = self.b.weight

        dist = ((F - G)**2).sum()
        F_flat = torch.reshape(F, (F.size(0), -1))
        G_flat = torch.reshape(G, (G.size(0), -1))
        cos_angle = ((F_flat*G_flat).sum(1))/torch.sqrt(((F_flat**2).sum(1))*((G_flat**2).sum(1)))     
        angle = (180.0/np.pi)*(torch.acos(cos_angle).mean().item())

        return dist, angle
    
    def weight_b_train(self, y, nb_iter, optimizer, sigma, arg_return = False):

        if self.alg == 1:        
            #noise_tab = []
            #dy_tab = []
            #dr_tab = []

            for iter in range(1, nb_iter + 1):
                y_temp, r_temp = self(y, back = True)
                noise = sigma*torch.randn_like(y)
                y_noise, r_noise = self(y + noise, back = True)
                dy = (y_noise - y_temp)
                dr = (r_noise - r_temp)
                
                #*****************************************************#
                loss_b = -(noise*dr).view(dr.size(0), -1).sum(1).mean()
                #*****************************************************#

                optimizer.zero_grad()
                loss_b.backward()
                optimizer.step()
               
                #noise_tab.append(noise.detach())
                #dy_tab.append(dy.detach())
                #dr_tab.append(dr.detach())        

            #renormalize once per sample
            self.weight_b_normalize(noise, dy, dr)
            #noise_tab = torch.stack(noise_tab, dim=0)
            #dy_tab = torch.stack(dy_tab, dim=0)
            #dr_tab = torch.stack(dr_tab, dim=0)
            #self.weight_b_normalize(noise_tab, dy_tab, dr_tab)

        elif self.alg == 2:
            for iter in range(1, nb_iter + 1):
                y_temp, r_temp = self(y, back = True)
                noise = sigma*torch.randn_like(y)
                y_noise, r_noise = self(y + noise, back = True)
                dy = (y_noise - y_temp)
                dr = (r_noise - r_temp)
              
                #**************************************# 
                noise_y = sigma*torch.randn_like(y_temp)
                r_noise_y = self.bb(y, y_temp + noise_y)
                dr_y = (r_noise_y - r_temp)
                loss_b = -2*(noise*dr).view(dr.size(0), -1).sum(1).mean() + (dr_y**2).view(dr_y.size(0), -1).sum(1).mean() 
                #**************************************#
                optimizer.zero_grad()
                loss_b.backward()
                optimizer.step()
     
        if arg_return:
            return loss_b

 
class globalNet(nn.Module):
    def __init__(self, args):
        super(globalNet, self).__init__()

        #MNIST        
        size = 28
        args.C = [1] + args.C

        layers = nn.ModuleList([])


        if not args.sigmapi:
            for i in range(len(args.C) - 1):
                layers.append(layer_conv(args.C[i], args.C[i + 1], args))
                size = int(np.floor((size - 5)/2 + 1))
            
            layers.append(layer_fc((size**2)*args.C[-1], 10, args, last_layer = True))
        else:
            for i in range(len(args.C) - 1):
                layers.append(layer_sigmapi_conv(args.C[i], args.C[i + 1], args))
                size = int(np.floor((size - 5)/2 + 1))
            
            layers.append(layer_sigmapi_fc((size**2)*args.C[-1], 10, args, last_layer = True))
    

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

    def weight_f_train(self, y, r, t, id_layer, optimizer):        
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


if __name__ == '__main__':
    
    BASE_PATH = createPath(args)
    command_line = ' '.join(sys.argv) 
    createHyperparameterfile(BASE_PATH, command_line, args)

    net = globalNet(args)
    net.to(device)

    print(net)

    #for name, param in net.named_parameters():
    #    print(param.size())

    if args.action == 'train': 
        if args.sym:
            net.weight_b_sym() 
        
        #Initialize optimizers for forward and backward weights
        optim_params_f = []
        optim_params_b = []

        for i in range(len(net.layers)):
            optim_params_f.append({'params': net.layers[i].f.parameters(), 'lr': args.lr_f})
            
        for i in range(len(net.layers) - 1):
            optim_params_b.append({'params': net.layers[i + 1].b.parameters(), 'lr': args.lr_b})

        optimizer_f = torch.optim.SGD(optim_params_f, momentum = 0.9) 
        optimizer_b = torch.optim.SGD(optim_params_b, momentum = 0.9)

        net.train()
        train_loss = 0
        correct = 0
        total = 0
        
        train_acc = []       
 
        for _ in range(args.epochs): 
            for batch_idx, (data, target) in enumerate(train_loader):

                data, target = data.to(device), target.to(device)
                
                #****FEEDBACK WEIGHTS****#
                if not args.sym:                
                    y = net.layers[0](data).detach() 
                    for id_layer in range(len(net.layers) - 1):                     
                        net.layers[id_layer + 1].weight_b_train(y, args.iter, optimizer_b, args.noise[id_layer])
                        #go to the next layer
                        y = net.layers[id_layer + 1](y).detach()

                #****FORWARD WEIGHTS****#
                y, r = net(data, ind = len(net.layers))

                pred = torch.exp(net.logsoft(y)) 
                target = F.one_hot(target, num_classes=10).float()
 
                t = y + args.beta*(target - pred)

                for id_layer in range(len(net.layers)):        
                    y, r, t, loss_f = net.weight_f_train(y, r, t, id_layer, optimizer_f)        
                    if id_layer == 0:
                        loss = loss_f
                    
                train_loss += loss.item()
                _, predicted = pred.max(1)
                _, targets = target.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
                             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
            train_acc.append(100.*correct/total)
            results = {'train_acc' : train_acc}
            outfile = open(os.path.join(BASE_PATH, 'results'), 'wb')
            pickle.dump(results, outfile)
            outfile.close() 

    #Coding the learning procedure for the feedback weights
    elif args.action == 'test':
              
        optim_params_b = []
            
        for i in range(len(net.layers) - 1):
            optim_params_b.append({'params': net.layers[i + 1].b.parameters(), 'lr': args.lr_b})

        optimizer_b = torch.optim.SGD(optim_params_b, momentum = 0.9)
        
        weight_tab = []
        loss_tab = []

        '''
        for id_layer in range(len(net.layers) - 1):
            weight_tab.append({'dist_weight' : [], 'angle_weight' : []})
            loss_tab.append([])
        '''
        
        weight_tab = {'dist_weight' : [], 'angle_weight' : []}        

        for batch_iter in range(args.epochs):
            _, (data, target) = next(enumerate(train_loader))             
            data, target = data.to(device), target.to(device)
                                       
            y = net.layers[0](data).detach()
            print('\n Batch iteration {}'.format(batch_iter + 1))
            
            for id_layer in range(len(net.layers) - 1):  
                if id_layer == len(net.layers) - 3:
                    loss_b = net.layers[id_layer + 1].weight_b_train(y, args.iter, optimizer_b, args.noise[id_layer])
     
                    dist_weight, angle_weight = net.layers[id_layer + 1].compute_dist_angle(y)

                    '''
                    loss_tab[id_layer].append(loss_b)
                    weight_tab[id_layer]['dist_weight'].append(dist_weight)
                    weight_tab[id_layer]['angle_weight'].append(angle_weight)
                    '''
                    
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

            #results = {'weight_tab' : weight_tab, 'loss_tab' : loss_tab}
            results = {'weight_tab' : weight_tab}
            outfile = open(os.path.join(BASE_PATH, 'results'), 'wb')
            pickle.dump(results, outfile)
            outfile.close() 

    #testing smallNet_benchmark
    '''
    _, (x, _) = next(enumerate(train_loader))
    
    x = x.to(device)
    net = smallNet_benchmark()
    net.to(device) 
    out = net(x)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9)
    for epochs in range(1, args.epochs + 1):
        train(epochs)
        test(epochs)

    print('All right!') 
    '''
