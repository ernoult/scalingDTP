# coding=utf-8
import argparse

from plotFunctions import *
from tools import *
from models import *

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


train_loader, test_loader = createDataset(args)


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


if __name__ == '__main__':

    if args.path is not None:    
        BASE_PATH = createPath(args)
        command_line = ' '.join(sys.argv) 
        createHyperparameterfile(BASE_PATH, command_line, seed, args)


    if args.dataset == 'mnist':
        net = smallNet(args)

    elif args.dataset == 'cifar10':
        net = VGG(args)
    
    net = net.to(device)
    print(net)    

    if args.action[0] == 'train': 
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        optimizers = createOptimizers(net, args, forward = True)
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
                pred, loss = train_batch(args, net, data, optimizers, target, criterion)

                train_loss += loss.item()
                _, predicted = pred.max(1)
                targets = target
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                #progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Train Acc: %.3f%% (%d/%d)'% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
              
            train_acc.append(100.*correct/total)
            test_acc_temp = test(net, test_loader, device)
            test_acc.append(test_acc_temp)

            if args.path is not None:
                results = {'train_acc' : train_acc, 'test_acc' : test_acc}
                outfile = open(os.path.join(BASE_PATH, 'results'), 'wb')
                pickle.dump(results, outfile)
                outfile.close()

            if (args.dataset == 'mnist') and (train_acc[-1] < 80): exit()
            elif (args.dataset == 'cifar10') and (train_acc[-1] < 30): exit()                 

    elif args.action[0] == 'test':
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
           
        optimizers = createOptimizers(net, args)
        
        weight_tab = []
        loss_tab = []

        weight_tab = {'dist_weight' : [], 'angle_weight' : []}        

        for batch_iter in range(args.epochs):
            print('\n Batch iteration {}'.format(batch_iter + 1))
            _, (data, _) = next(enumerate(train_loader))             
            data = data.to(device)

            weight_tab = train_batch(args, net, data, optimizers, tabs = weight_tab, batch_iter = batch_iter)
               

            if args.path is not None:
                results = {'weight_tab' : weight_tab}
                outfile = open(os.path.join(BASE_PATH, 'results'), 'wb')
                pickle.dump(results, outfile)
                outfile.close() 

        
    elif args.action[0] == 'debug':

        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        optimizer_f, optimizer_b = createOptimizers(net, args, forward = True)

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
        '''
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
        '''
              

