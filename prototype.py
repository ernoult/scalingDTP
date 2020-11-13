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



parser = argparse.ArgumentParser(description='Testing idea of Yoshua')


parser.add_argument('--in_size', type=int, default=784, help='input dimension (default: 784)')   
parser.add_argument('--out_size', type=int, default=512, help='output dimension (default: 512)')   
parser.add_argument('--epochs', type=int, default=15, help='number of epochs to train feedback weights(default: 15)') 
parser.add_argument('--batch-size', type=int, default=128, help='batch dimension (default: 128)')   
parser.add_argument('--device-label', type=int, default=0, help='device (default: 1)')   
parser.add_argument('--noise', type=float, default=0.05, help='noise level (default: 0.05)')   
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')   
parser.add_argument('--lamb', type=float, default=0.01, help='regularization parameter (default: 0.01)')   
parser.add_argument('--seed', default=False, action='store_true',help='fixes the seed to 1')
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


transforms=[torchvision.transforms.ToTensor(),ReshapeTransform((-1,))]


train_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST(root='./data', train=True, download=True,
                         transform=torchvision.transforms.Compose(transforms),
                         target_transform=ReshapeTransformTarget(10)),
batch_size = args.batch_size, shuffle=True)

if args.device_label >= 0:    
    device = torch.device("cuda:"+str(args.device_label))
else:
    device = torch.device("cpu")

class prototype(nn.Module):
    def __init__(self, args):
        super(prototype, self).__init__()
        self.f = nn.Linear(args.in_size, args.out_size)
        self.b = nn.Linear(args.out_size, args.in_size)
    

    def ff(self, x):
        y = self.f(torch.tanh(x))
        #y = self.f(x)
        return y

    def bb(self, y):
        r = self.b(torch.tanh(y))
        #r = self.b(y)
        return r
        

    def forward(self, x):
        y = self.ff(x)
        r = self.bb(y)
        
        return y, r

def compute_dist_jacobians(net, x, y):
    jac_F = torch.autograd.functional.jacobian(net.ff, x) 
    jac_F = torch.transpose(torch.diagonal(jac_F, dim1=0, dim2=2), 0, 2)
    jac_G = torch.autograd.functional.jacobian(net.bb, y) 
    jac_G = torch.transpose(torch.diagonal(jac_G, dim1=0, dim2=2), 0, 2)
   
    jac_G = torch.transpose(jac_G, 1, 2) 
    #print(jac_F.size())
    #print(jac_G.size())
    dist = ((jac_F - jac_G)**2).sum(2).sum(1).mean()
    #print(dist)    
    return dist

if __name__ == '__main__':    

    #Testing prototype
    '''    
    _, (x, _) = next(enumerate(train_loader))     
    x = x.to(device)
    net = prototype(args)
    net.to(device)
    y, r = net(x)
   
    #testing jacobian computation on a simple case 
    compute_dist_jacobians(net, x, y)    

    jac = torch.autograd.functional.jacobian(net.ff, x) 
    jac = torch.transpose(torch.diagonal(jac, dim1 = 0, dim2 = 2), 0, 2)
    print('Jacobian size: {}'.format(jac.size()))
    #print(jac)
    print(jac[0, :] == torch.transpose(net.f.weight, 0, 1))
    print('Done!')
    '''

    #Coding the learning procedure

    _, (x, _) = next(enumerate(train_loader))     
    x = x.to(device)
    net = prototype(args)
    net.to(device)


    optimizer_b = torch.optim.SGD([{'params': net.b.parameters()}], lr=args.lr, momentum=9e-1)
    
    
    for iter in range(1, args.epochs + 1):
        y, r = net(x)
        noise = args.noise*torch.randn_like(x)
        y_noise, r_noise = net(x + noise)
        dy = (y_noise - y)
        dr = (r_noise - r)
        loss_b =(1/args.noise)*(-args.lamb*(dr**2).sum(1) + ((dy**2).sum(1) - (noise*dr).sum(1))**2).mean()
        #loss_b = ((1/args.noise)*((dy**2).sum(1) - (noise*dr).sum(1))**2).mean()
        dist_jac = compute_dist_jacobians(net, x, y)
        print('Feedback loss at step {}: {:5f}'.format(iter, loss_b))
        print('Distance between jacobians at step {}: {:5f}'.format(iter, dist_jac))
        optimizer_b.zero_grad()
        loss_b.backward(retain_graph = True)
        optimizer_b.step()
     
    print('Done!')
