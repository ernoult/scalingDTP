import os, sys
import stat
import pickle
import datetime
from shutil import copyfile
import copy
import time

'''
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
'''

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

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
