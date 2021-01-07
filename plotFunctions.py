import numpy as np
import matplotlib.pyplot as plt
import os, sys
import pickle
import torch
from matplotlib.ticker import MaxNLocator

fontsize = 12
linewidth = 5

def plot_results(results):
    
    #Loss stuff
        
    epochs = len(results['loss_tab'][1])
    '''
    print(results['loss_tab'][1])
    fig1 = plt.figure(figsize=(5, 4))
    plt.rcParams.update({'font.size': fontsize}) 
    plt.plot(np.linspace(1, epochs, epochs), results['loss_tab'][1], linewidth=2.5, alpha=0.8)
    plt.xlabel('Batch iterations')
    plt.ylabel('Loss of feedback weights')
    plt.grid()
    plt.subplots_adjust(hspace = 0.5)
    fig1.tight_layout()    
    fig1.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    '''

    #Weight stuff
    fig3 = plt.figure(figsize=(8, 4))
    plt.rcParams.update({'font.size': fontsize}) 
    plt.subplot(121)
    plt.plot(np.linspace(1, epochs, epochs), results['weight_tab'][1]['angle_weight'], linewidth=2.5, alpha=0.8)
    plt.xlabel('Batch iterations')
    plt.ylabel('Angle between ' + r'$w_f^\top$'+ ' and ' + r'$w_b$' + r' $ (\circ)$')
    plt.grid()
    fig3.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.subplot(122)
    plt.plot(np.linspace(1, epochs, epochs), results['weight_tab'][1]['dist_weight'], linewidth=2.5, alpha=0.8)
    plt.xlabel('Batch iterations')
    plt.ylabel('Distance between ' + r'$w_f^\top$'+ ' and ' + r'$w_b$')
    plt.grid()
    plt.subplots_adjust(hspace = 0.5)
    fig3.tight_layout() 
    fig3.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

if __name__ == '__main__':
    BASE_PATH = os.getcwd() + '/results' 
    infile = open(BASE_PATH,'rb')
    results_dict = pickle.load(infile)
    infile.close()
    #print(results_dict['weight_tab'][1]['angle_weight'])
    plot_results(results_dict)
    plt.show()
