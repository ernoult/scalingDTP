import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



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
        if hasattr(self, 'b'):
            r = self.unpool(y, ind, output_size = x.size())
            r = self.b(self.rho(r), output_size = x.size())
            return r 
        else:
            return None

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
        angle = (180.0/np.pi)*(torch.acos(cos_angle).mean().item())
        
        return dist, angle

    def weight_b_train(self, y, optimizer, arg_return = False):
        
        nb_iter = self.iter
        sigma = self.noise
            
        for iter in range(1, nb_iter + 1):
            y_temp, (r_temp, ind) = self(y, back = True)
            noise = sigma*torch.randn_like(y)
            y_noise, (r_noise, ind_noise) = self(y + noise, back = True)
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
        '''
        for name, p in net.named_parameters():
            if p.grad is not None:
                print(name + ' has mean gradient {}'.format(p.grad.mean()))
        '''
        
        return loss_f

    def propagateError(self, r_tab, t):
        r = r_tab[0]
        ind = r_tab[1]
        delta = self.bb(r_tab[0], t, r_tab[1]) - r_tab[0]

        return delta

class layer_sigmapi_convpool(nn.Module):
    def __init__(self, args, in_channels, out_channels, activation, iter=None, noise=None):
        super(layer_sigmapi_convpool, self).__init__()
        self.f = nn.Conv2d(in_channels, out_channels, 3, stride = 1, padding = 1)
        self.pool = nn.MaxPool2d(2, stride = 2, return_indices = True)            
        
        if iter is not None:
            b = nn.ModuleList([nn.ConvTranspose2d(out_channels, in_channels, 3, stride = 1, padding = 1), 
                                nn.ConvTranspose2d(out_channels, in_channels, 3, stride = 1, padding = 1)])
            self.b = b
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
        #******************#
    
    def ff(self, x, ret_ind = False):
        y, ind = self.pool(self.rho(self.f(x)))
        
        if ret_ind:
            return y, ind
        else:
            return y

    def bb(self, x, y, ind):
        if hasattr(self, 'b'):
            r = self.unpool(y, ind, output_size = x.size())
            r = self.b[0](self.rho(r), output_size = x.size())*self.b[1](self.rho(r), output_size = x.size())
            return r 
        else:
            return None

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

        
        F_flat = torch.reshape(F, (F.size(0), -1))
        G_flat = torch.reshape(G, (G.size(0), -1))
        cos_angle = ((F_flat*G_flat).sum(1))/torch.sqrt(((F_flat**2).sum(1))*((G_flat**2).sum(1)))     
        angle = (180.0/np.pi)*(torch.acos(cos_angle).mean().item())
        
        return dist, angle

    def weight_b_train(self, y, optimizer, arg_return = False):
        
        nb_iter = self.iter
        sigma = self.noise
            
        for iter in range(1, nb_iter + 1):
            y_temp, (r_temp, ind) = self(y, back = True)
            noise = sigma*torch.randn_like(y)
            y_noise, (r_noise, ind_noise) = self(y + noise, back = True)
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
        '''
        for name, p in net.named_parameters():
            if p.grad is not None:
                print(name + ' has mean gradient {}'.format(p.grad.mean()))
        '''
        
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



class VGG(nn.Module):
    def __init__(self, args):
        super(VGG, self).__init__()

        #CIFAR-10       
        size = 32
        args.C = [3] + args.C

        layers = nn.ModuleList([])

        layers.append(layer_convpool(args, args.C[0], args.C[1], args.activation))
        size = int(np.floor(size/2))

        if args.sigmapi:
            for i in range(len(args.C) - 2):
                layers.append(layer_sigmapi_convpool(args, args.C[i + 1], args.C[i + 2], 
                                        args.activation, args.iter[i], args.noise[i]))
                
                size = int(np.floor(size/2))
        
        else:
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




