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
from utils import * 

parser = argparse.ArgumentParser(description='Testing idea of Yoshua')


parser.add_argument('--in_size', type=int, default=784, help='input dimension (default: 784)')   
parser.add_argument('--out_size', type=int, default=512, help='output dimension (default: 512)')   
parser.add_argument('--in_channels', type=int, default=1, help='input channels (default: 1)')   
parser.add_argument('--out_channels', type=int, default=128, help='output channels (default: 128)')   
parser.add_argument('--epochs', type=int, default=15, help='number of epochs to train feedback weights(default: 15)') 
parser.add_argument('--iter', type=int, default=20, help='number of iterationson feedback weights per batch samples (default: 20)') 
parser.add_argument('--batch-size', type=int, default=128, help='batch dimension (default: 128)')   
parser.add_argument('--device-label', type=int, default=0, help='device (default: 1)')   
parser.add_argument('--noise', type=float, default=0.05, help='noise level (default: 0.05)')   
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')   
parser.add_argument('--lamb', type=float, default=0.01, help='regularization parameter (default: 0.01)')   
parser.add_argument('--seed', default=False, action='store_true',help='fixes the seed to 1 (default: False)')
parser.add_argument('--jacobian', default=False, action='store_true',help='compute jacobians (default: False)')
parser.add_argument('--conv', default=False, action='store_true',help='select the conv archi (default: False)')
parser.add_argument('--C', nargs = '+', type=int, default=[128, 512], help='tab of channels (default: [128, 512])')
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


if (args.conv):
    transforms=[torchvision.transforms.ToTensor()]
else:
    transforms=[torchvision.transforms.ToTensor(),ReshapeTransform((-1,))]


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


class layer_fc(nn.Module):
    def __init__(self, in_size, out_size, last_layer = False):
        super(layer_fc, self).__init__()
        self.f = nn.Linear(in_size, out_size)
        self.b = nn.Linear(out_size, in_size)
        self.last_layer = last_layer

    def ff(self, x):
        #y = self.f(torch.tanh(x))
        if self.last_layer:
            x_flat = x.view(x.size(0), - 1)
            y = self.f(x_flat)
        else:
            y = self.f(x)
        return y

    def bb(self, x, y):
        #r = self.b(torch.tanh(y))
        r = self.b(y)
        
        if self.last_layer:
            r = r.view(x.size())

        return r        

    def forward(self, x):
        y = self.ff(x)
        r = self.bb(x, y)
        
        y = y.detach()
        y.requires_grad = True

        return y, r


    def weight_b_normalize(self, dx, dy, dr):
                     
        factor = ((dy**2).sum(1))/((dx*dr).view(dx.size(0), -1).sum(1))
        factor = factor.mean()
        #factor = 0.5*factor

        with torch.no_grad():
            self.b.weight.data = factor*self.b.weight.data

class layer_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(layer_conv, self).__init__()
        self.f = nn.Conv2d(in_channels, out_channels, 5, stride = 2)
        self.b = nn.ConvTranspose2d(out_channels, in_channels, 5, stride = 2)
    

    def ff(self, x):
        y = F.relu(self.f(x))
        #y = self.f(x)
        return y

    def bb(self, x, y):
        #r = F.relu(self.b(y))
        r = F.relu(self.b(y, output_size = x.size()))
        return r 

    def forward(self, x):
        y = self.ff(x)
        r = self.bb(x, y)
        y = y.detach()
        y.requires_grad = True
        return y, r


    def weight_b_normalize(self, dx, dy, dr):
        
        #first technique: same normalization for all out fmaps        
        
        dy = dy.view(dy.size(0), -1)
        dx = dx.view(dx.size(0), -1)
        dr = dr.view(dr.size(0), -1)
       

        #print(dy.size())
        #print(dx.size())
        #print(dr.size())
 
        factor = ((dy**2).sum(1))/((dx*dr).sum(1))
        factor = factor.mean()
        #factor = 0.5*factor

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

def compute_jacobians(net, x, y):
    jac_F = torch.autograd.functional.jacobian(net.ff, x) 
    jac_F = torch.transpose(torch.diagonal(jac_F, dim1=0, dim2=2), 0, 2)
    jac_G = torch.autograd.functional.jacobian(net.bb, y) 
    jac_G = torch.transpose(torch.diagonal(jac_G, dim1=0, dim2=2), 0, 2)
   
    jac_G = torch.transpose(jac_G, 1, 2) 
 

    return jac_F, jac_G

def compute_dist_angle(F, G, jac = False):
    
    if jac:    
        dist = ((F - G)**2).sum(2).sum(1).mean()
    else:
        dist = ((F - G)**2).sum()

    F_flat = torch.reshape(F, (F.size(0), -1))
    G_flat = torch.reshape(G, (G.size(0), -1))
    cos_angle = ((F_flat*G_flat).sum(1))/torch.sqrt(((F_flat**2).sum(1))*((G_flat**2).sum(1)))     
    angle = (180.0/np.pi)*(torch.acos(cos_angle).mean().item())

    return dist, angle
  

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


class globalNet(nn.Module):
    def __init__(self, args):
        super(globalNet, self).__init__()

        #MNIST        
        size = 28
        args.C = [1] + args.C

        layers = nn.ModuleList([])
        for i in range(len(args.C) - 1):
            layers.append(layer_conv(args.C[i], args.C[i + 1]))
            size = int(np.floor((size - 5)/2 + 1))
        
        layers.append(layer_fc((size**2)*args.C[-1], 10, last_layer = True))

        self.layers = layers

    def forward(self, x):
        s = x
        y = []
        r = []

        for i in range(len(self.layers)):
            y_temp, r_temp = self.layers[i](s)      
            y.append(y_temp)
            if i > 0:
                r.append(r_temp)
            s = y_temp

        return y, r
 

if __name__ == '__main__':


    #Testing globalNet
    '''    
    net = globalNet(args)
    net.to(device)
    
    
    _, (data, target) = next(enumerate(train_loader))         
    
    data = data.to(device)    

    #Initialize optimizers for forward and backward weights
    
    optim_params_f = []
    optim_params_b = []

    for i in range(len(net.layers)):
        optim_params_f.append({'params': net.layers[i].f.parameters(), 'lr': args.lr})
        
    for i in range(len(net.layers) - 1):
        optim_params_b.append({'params': net.layers[i + 1].b.parameters(), 'lr': args.lr})

    optimizer_f = torch.optim.SGD(optim_params_f, momentum = 0.9) 
    optimizer_b = torch.optim.SGD(optim_params_b, momentum = 0.9)

    #forward pass

    y, r = net(data)
    
     
    #for i in range(len(y)):
    #    print('Layer y {}: {}'.format(i + 1, y[i].size()))
    #    print('Layer r {}: {}'.format(i + 1, r[i].size()))
    
    
    #train feedback weights
    
    for id_layer in range(len(net.layers) - 1):
        print('Layer {}...'.format(id_layer + 2))
        
        for iter in range(1, args.iter + 1):
            if (iter % 10 == 0):
                print('Iteration {}'.format(iter))

            y_temp, r_temp = net.layers[id_layer + 1](y[id_layer])
            noise = args.noise*torch.randn_like(y[id_layer])
            
            noisy_input = y[id_layer] + noise
                                  
            y_noise, r_noise = net.layers[id_layer + 1](noisy_input)

            dy = (y_noise - y_temp)
            dr = (r_noise - r_temp)
           
            loss_b = -(noise*dr).view(dr.size(0), -1).sum(1).mean()
            

            optimizer_b.zero_grad()
               
            if iter < args.iter:
                loss_b.backward(retain_graph = True)
            else:
                loss_b.backward()

            optimizer_b.step()
       
        #WATCH OUT: renormalize once per sample
        net.layers[id_layer + 1].weight_b_normalize(noise, dy, dr)        
        
        if id_layer < len(net.layers) - 2:
            dist_weight, angle_weight = compute_dist_angle(net.layers[id_layer + 1].f.weight, net.layers[id_layer + 1].b.weight) 
        else:
            dist_weight, angle_weight = compute_dist_angle(net.layers[id_layer + 1].f.weight, net.layers[id_layer + 1].b.weight.t())

        print('Distance between weights: {:.2f}'.format(dist_weight))
        print('Weight angle: {:.2f} deg'.format(angle_weight))         

    print('Good!')    
    '''
    #Testing prototype
    
    '''           
    _, (x, _) = next(enumerate(train_loader))     
    
    #x = torch.randn(args.batch_size, 1, 28, 28)
    x = x.to(device)

    if args.conv:
        net = layer_conv(args.in_channels, args.out_channels)
    else:
        net = layer_fc(args.in_size, args.out_size)
    
    net.to(device)
    y, r = net(x)
   
    #test normalization

    noise = args.noise*torch.randn_like(x)
    y_noise, r_noise = net(x + noise)

    net.weight_b_normalize(noise, y_noise - y, r_noise - r)        
 
    #print(x.size())
    #print(y.size())
    #print(r.size())    

    print('Done!')
    
    print(net.f.weight.size())
    print(net.b.weight.size())
    '''

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

    #testing jacobian computation on a simple case 
 
    #compute_dist_jacobians(net, x, y)
    
    '''
    jac = torch.autograd.functional.jacobian(net.ff, x) 
    jac = torch.transpose(torch.diagonal(jac, dim1 = 0, dim2 = 2), 0, 2)
    print(torch.all(torch.eq(jac[0, :], torch.transpose(net.f.weight, 0, 1)))) 

    jac_2 = torch.autograd.functional.jacobian(net.bb, y) 
    jac_2 = torch.transpose(torch.diagonal(jac_2, dim1 = 0, dim2 = 2), 0, 2)
    print(torch.all(torch.eq(jac_2[0, :], torch.transpose(net.b.weight, 0, 1)))) 
    '''
    #print('Done!')
    
    #Coding the learning procedure for the feedback weights 
           
    BASE_PATH = createPath(args) 
    createHyperparameterfile(BASE_PATH, args)
    
    if args.conv:
        net = layer_conv(args.in_channels, args.out_channels)
    else:
        net = layer_fc(args.in_size, args.out_size)        

    net.to(device)

    optimizer_b = torch.optim.SGD([{'params': net.b.parameters()}], lr=args.lr, momentum=9e-1)   
    angle_jac_tab, dist_jac_tab, loss_tab, angle_weight_tab, dist_weight_tab = [], [], [], [], []      
  
    #WATCH OUT: only ONE data point!
    _, (x, _) = next(enumerate(train_loader))     
    x = x.to(device)
 
    for iter_x in range(1, args.epochs + 1):
        #_, (x, _) = next(enumerate(train_loader))     
        #x = x.to(device)

        for iter in range(1, args.iter + 1):
            y, r = net(x)
            noise = args.noise*torch.randn_like(x)
            y_noise, r_noise = net(x + noise)
            dy = (y_noise - y)
            dr = (r_noise - r)
           
            #LOSS 1 
            #loss_b =(1/args.noise)*(-args.lamb*(dr**2).sum(1) + ((dy**2).sum(1) - (noise*dr).sum(1))**2).mean()
            
            #LOSS 2
            #loss_b =-(args.lamb/(args.noise**2))*(dr**2).sum(1).mean() + (1/(args.noise**4))*( ((dy**2).sum(1) - (noise*dr).sum(1))**2).mean()
            

            #LOSS 3
            #net.weight_b_normalize(noise, dy, dr)
            #loss_b = -(noise*dr).sum(1).mean()
            
            #LOSS 4
            #net.weight_b_normalize(noise, dy, dr)
            #loss_b = -(((noise*dr).sum(1))**2).mean()

            #LOSS 5
            loss_b = -(noise*dr).view(dr.size(0), -1).sum(1).mean()
            

            optimizer_b.zero_grad()
               
            if iter < args.iter:
                loss_b.backward(retain_graph = True)
            else:
                loss_b.backward()

            optimizer_b.step()
       
        #WATCH OUT: renormalize once per sample
        net.weight_b_normalize(noise, dy, dr)        
 
        loss_tab.append(loss_b)
        results_dict = {'loss': loss_tab}
        print('\n Batch {} ({} trials per batch): \n'.format(iter_x, args.iter))
        print('Feedback loss: {:.2f}'.format(loss_b))
        
        if args.jacobian:
            jac_f, jac_b = compute_jacobians(net, x, y)
            dist_jac, angle_jac = compute_dist_angle(jac_f, jac_b, jac = True)
            dist_jac_tab.append(dist_jac)
            angle_jac_tab.append(angle_jac)
            results_dict_jac = {'dist_jac': dist_jac_tab, 'angle_jac': angle_jac_tab}
            results_dict.update(results_dict_jac)
            print('Distance between jacobians: {:.2f}'.format(dist_jac))
            print('Jacobian angle: {:.2f} deg'.format(angle_jac))       
        if args.conv:
            dist_weight, angle_weight = compute_dist_angle(net.f.weight, net.b.weight) 
        else:
            dist_weight, angle_weight = compute_dist_angle(net.f.weight, net.b.weight.t())
        
        dist_weight_tab.append(dist_weight)
        angle_weight_tab.append(angle_weight)  
        results_dict_weight = {'dist_weight': dist_weight_tab, 'angle_weight': angle_weight_tab}
        results_dict.update(results_dict_weight)
        print('Distance between weights: {:.2f}'.format(dist_weight))
        print('Weight angle: {:.2f} deg'.format(angle_weight))
 
        outfile = open(os.path.join(BASE_PATH, 'results'), 'wb')
        pickle.dump(results_dict, outfile)
        outfile.close()
 
    print('Done!')
    plot_results(results_dict) 
    plt.show()
