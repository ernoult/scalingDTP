import numpy as np
import matplotlib.pyplot as plt
import os, sys
import pickle
import torch
from matplotlib.ticker import MaxNLocator

fontsize = 12
linewidth = 5

def plot_results(results):
    
    if 'weight_tab' in results:
        
        epochs = len(results['weight_tab']['dist_weight'])

        #Weight stuff
        fig3 = plt.figure(figsize=(8, 4))
        plt.rcParams.update({'font.size': fontsize}) 
        plt.subplot(121)
        plt.plot(np.linspace(1, epochs, epochs), results['weight_tab']['angle_weight'], linewidth=2.5, alpha=0.8)
        plt.xlabel('Batch iterations')
        plt.ylabel('Angle between ' + r'$w_f^\top$'+ ' and ' + r'$w_b$' + r' $ (\circ)$')
        plt.grid()
        fig3.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.subplot(122)
        plt.plot(np.linspace(1, epochs, epochs), results['weight_tab']['dist_weight'], linewidth=2.5, alpha=0.8)
        plt.xlabel('Batch iterations')
        plt.ylabel('Distance between ' + r'$w_f^\top$'+ ' and ' + r'$w_b$')
        plt.grid()
        plt.subplots_adjust(hspace = 0.5)
        fig3.tight_layout() 
        fig3.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    elif 'train_acc' in results:

        epochs = len(results['train_acc'])

        fig1 = plt.figure(figsize=(6, 4))
        plt.rcParams.update({'font.size': fontsize}) 
        plt.plot(np.linspace(1, epochs, epochs), results['train_acc'], linewidth=2.5, alpha=0.8)
        plt.plot(np.linspace(1, epochs, epochs), results['test_acc'], linewidth=2.5, alpha=0.8)
        plt.xlabel('Epochs')
        plt.ylabel('Train accuracy (%)')
        plt.grid()
        fig1.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        fig1.tight_layout()

        epochs = len(results['angle'][0])
        fig2 = plt.figure(figsize=(8, 4))
        plt.rcParams.update({'font.size': fontsize})
        plt.subplot(121)
        plt.plot(np.linspace(0, epochs - 1, epochs), results['angle'][0], linewidth=2.5, alpha=0.8, label = 'CONV')
        plt.plot(np.linspace(0, epochs - 1, epochs), results['angle'][1], linewidth=2.5, alpha=0.8, label = 'FC')
        plt.xlabel('Epochs')
        plt.ylabel('Weight angle (deg)')
        plt.grid()
        fig2.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.subplot(122)
        plt.plot(np.linspace(0, epochs - 1, epochs), results['dist'][0], linewidth=2.5, alpha=0.8, label = 'CONV')
        plt.plot(np.linspace(0, epochs - 1, epochs), results['dist'][1], linewidth=2.5, alpha=0.8, label = 'FC')
        plt.xlabel('Epochs')
        plt.ylabel('Relative weight distance')
        plt.grid()
        plt.legend(loc = 'best')
        fig2.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        fig2.tight_layout() 

if __name__ == '__main__':
    BASE_PATH = os.getcwd() + '/results' 
    infile = open(BASE_PATH,'rb')
    results = pickle.load(infile)
    infile.close()
    if 'train_acc' in results:
        print('Final train accuracy: {}'.format(results['train_acc'][-1]))
        print('Final test accuracy: {}'.format(results['test_acc'][-1]))
    plot_results(results)
    plt.show()
