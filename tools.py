import os, sys
import stat
import pickle
import datetime
from shutil import copyfile
import torch
import copy
import time
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def createPath(args):

    if args.path is None:
        if args.action == 'train':
            BASE_PATH = os.getcwd() + '/train_forward' 
        elif args.action == 'test':
            BASE_PATH = os.getcwd() + '/train_feedback' 
        
        if not os.path.exists(BASE_PATH):
            os.mkdir(BASE_PATH)

        BASE_PATH = BASE_PATH + '/' + datetime.datetime.now().strftime("%Y-%m-%d")
    
    else:
        BASE_PATH = os.getcwd() + '/' + args.path

    if not os.path.exists(BASE_PATH):
        os.mkdir(BASE_PATH)

    files = os.listdir(BASE_PATH)

    BASE_PATH_glob = BASE_PATH
    
    if not files:
        BASE_PATH = BASE_PATH + '/' + 'Trial-1'
    else:
        tab = []
        for names in files:
            if not names[-2] == '-':
                tab.append(int(names[-2] + names[-1]))
            else:    
                tab.append(int(names[-1]))
        
        BASE_PATH = BASE_PATH + '/' + 'Trial-' + str(max(tab)+1)                                
    
    os.mkdir(BASE_PATH) 
    filename = 'results'   
    
    #************************************#
    copyfile('plotFunctions.py', BASE_PATH + '/plotFunctions.py')
    
    if args.last_trial:
        copyfile('compute_stats.py', BASE_PATH_glob + '/compute_stats.py')    
    #************************************#

    return BASE_PATH

def createHyperparameterfile(BASE_PATH, command_line, seed, args):    
    
    command = 'python '+ command_line

    if args.seed is None:
        command += ' --seed ' + str(seed)
        

    hyperparameters = open(BASE_PATH + r"/hyperparameters.txt","w+") 
    L = ["List of hyperparameters " + "(" +  datetime.datetime.now().strftime("cuda" + str(args.device_label)+"-%Y-%m-%d") + ") \n",
        "- number of channels per conv layer: {}".format(args.C) + "\n",
        "- number of feedback optim steps per batch: {}".format(args.iter) + "\n",
        "- activation function: {}".format(args.activation) + "\n",
        "- noise: {}".format(args.noise) + "\n",
        "- algorithm used to train the feedback weights: {}".format(args.alg) + "\n",
        "- learning rate for forward weights: {}".format(args.lr_f) + "\n",
        "- learning rate for feedback weights: {}".format(args.lr_b) + "\n",
        "- batch size: {}".format(args.batch_size) + "\n",
        "- symmetric weight initialization: {}".format(args.sym) + "\n",
        "- number of epochs: {}".format(args.epochs) + "\n",
        "- seed: {}".format(seed) + "\n", 
        "\n To reproduce this simulation, type in the terminal:\n",
        "\n" + command + "\n"]

    hyperparameters.writelines(L) 
    hyperparameters.close()

    script = open(BASE_PATH + r"/reproduce_exp.sh", "w+")
    L = ["(cd " + os.getcwd() + ";" + command + ")"]
    script.writelines(L)
    script.close()
   
    script_name = BASE_PATH + '/reproduce_exp.sh'
    st = os.stat(script_name)
    os.chmod(script_name, st.st_mode | stat.S_IEXEC)    
 
'''
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
'''

def train_batch(args, net, data, optimizers, target = None, criterion = None, **kwargs):

    if args.action[0] == 'train':
        optimizer_f, optimizer_b = optimizers


        #****FEEDBACK WEIGHTS****#          
        y = net.layers[0](data).detach() 
        for id_layer in range(len(net.layers) - 1):                     
            net.layers[id_layer + 1].weight_b_train(y, optimizer_b)
            y = net.layers[id_layer + 1](y).detach()

        pred = torch.exp(net.logsoft(y)) 

        #*********FORWARD WEIGHTS********#
        y, r = net(data, ind_layer = len(net.layers))

        L = criterion(y.float(), target).squeeze()
        init_grads = torch.tensor([1 for i in range(y.size(0))], dtype=torch.float, device=y.device, requires_grad=True) 
        grads = torch.autograd.grad(L, y, grad_outputs=init_grads, create_graph = True)
        delta = -args.beta*grads[0]

        t = y + delta

        for id_layer in range(len(net.layers)):        
            
            loss_f = net.layers[-1 - id_layer].weight_f_train(y, t, optimizer_f)

            #compute previous targets         
            if (id_layer < len(net.layers) - 1):
                delta = net.layers[-1 - id_layer].propagateError(r, t)
                y, r = net(data, ind_layer = len(net.layers) - 1 - id_layer)
                t = (y + delta).detach()
            
            if id_layer == 0:
                loss = loss_f
        
        
        return pred, loss

    elif args.action[0] == 'test':
        optimizer_b = optimizers
        weight_tab = kwargs['tabs']
        batch_iter = kwargs['batch_iter']
        y = net.layers[0](data).detach()
        
        for id_layer in range(len(net.layers) - 1):  
            if (len(args.action) == 1) or id_layer == int(args.action[1]):
                loss_b = net.layers[id_layer + 1].weight_b_train(y, optimizer_b)

                if (batch_iter + 1) % 10  == 0: 
                    dist_weight, angle_weight = net.layers[id_layer + 1].compute_dist_angle(y)
                    
                    if id_layer < len(net.layers) - 2:
                        print('Conv layer ' + str(id_layer + 1) + ': ')
                    else:
                        print('FC layer:')  

                    print('Distance between weights: {:.2f}'.format(dist_weight))
                    print('Weight angle: {:.2f} deg'.format(angle_weight))  
                    weight_tab['dist_weight'].append(dist_weight)
                    weight_tab['angle_weight'].append(angle_weight)       

            #go to the next layer
            y = net.layers[id_layer + 1](y).detach()

        return weight_tab



def createOptimizers(net, args, forward = False):

    optim_params_b = []     
    for i in range(len(net.layers) - 1):
        optim_params_b.append({'params': net.layers[i + 1].b.parameters(), 'lr': args.lr_b[i]})
     
    optimizer_b = torch.optim.SGD(optim_params_b, momentum = 0.9)

    if forward: 
        optim_params_f = []
        for i in range(len(net.layers)):
            optim_params_f.append({'params': net.layers[i].f.parameters(), 'lr': args.lr_f})
        optimizer_f = torch.optim.SGD(optim_params_f, momentum = 0.9)
        return (optimizer_f, optimizer_b) 
    
    else:
        return optimizer_b




def test(net, test_loader, device):

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

def copy(y, ind_y):
    y_copy = []
    
    for i in range(len(y)):
        y_copy.append(y[i].clone())

    #WATCH OUT: detach previous node!
    y_copy[ind_y - 1] = y_copy[ind_y - 1].detach()    

    return y_copy

def createDataset(args):

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
        '''
        transform_train = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                                        torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                                                                                               std=(3*0.2023, 3*0.1994, 3*0.2010)) ])
        '''

        transform_train = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(0.5),
                                                          torchvision.transforms.RandomCrop(size=[32,32], padding=4, padding_mode='edge'),
                                                          torchvision.transforms.ToTensor(), 
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

    return train_loader, test_loader

