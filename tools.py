import os, sys
import pickle
import datetime
from shutil import copyfile
import copy

def createPath(args):

    BASE_PATH = os.getcwd() + '/learn_jacobian' 
    
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

def createHyperparameterfile(BASE_PATH, args):    

    hyperparameters = open(BASE_PATH + r"/hyperparameters.txt","w+") 
    L = ["List of hyperparameters " + "(" +  datetime.datetime.now().strftime("cuda" + str(args.device_label)+"-%Y-%m-%d") + ") \n",
        "- noise: {}".format(args.noise) + "\n",
        "- learning rate: {}".format(args.lr) + "\n",
        "- lambda: {}".format(args.lamb) + "\n",
        "- batch size: {}".format(args.batch_size) + "\n",
        "- number of epochs: {}".format(args.epochs) + "\n",
        "- fixed seed: {}".format(args.seed) + "\n"]


    hyperparameters.writelines(L) 
    hyperparameters.close()
