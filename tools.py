import os, sys
import pickle
import datetime
from shutil import copyfile
import copy
import time



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

    if args.action == 'train':
        BASE_PATH = os.getcwd() + '/train_forward' 
    elif args.action == 'test':
        BASE_PATH = os.getcwd() + '/train_feedback' 
    
    if not os.path.exists(BASE_PATH):
        os.mkdir(BASE_PATH)

    BASE_PATH = BASE_PATH + '/' + datetime.datetime.now().strftime("%Y-%m-%d")

    if not os.path.exists(BASE_PATH):
        os.mkdir(BASE_PATH)

    files = os.listdir(BASE_PATH)

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
    #************************************#

    return BASE_PATH

def createHyperparameterfile(BASE_PATH, command_line, args):    

    hyperparameters = open(BASE_PATH + r"/hyperparameters.txt","w+") 
    L = ["List of hyperparameters " + "(" +  datetime.datetime.now().strftime("cuda" + str(args.device_label)+"-%Y-%m-%d") + ") \n",
        "- noise: {}".format(args.noise) + "\n",
        "- algorithm used to train the feedback weights: {}".format(args.alg) + "\n",
        "- learning rate for forward weights: {}".format(args.lr_f) + "\n",
        "- learning rate for feedback weights: {}".format(args.lr_b) + "\n",
        "- batch size: {}".format(args.batch_size) + "\n",
        "- symmetric weight initialization: {}".format(args.sym) + "\n",
        "- number of epochs: {}".format(args.epochs) + "\n",
        "- fixed seed: {}".format(args.seed) + "\n", 
        "\n To reproduce this simulation, type in the terminal:\n",
        "\n" + command_line + "\n"]


    hyperparameters.writelines(L) 
    hyperparameters.close()
