import numpy as np
import matplotlib.pyplot as plt
import os, sys
import pickle
import torch

def plot_results(stat_w, stat_angle):   

    linewidth = 3
    epochs = np.shape(stat_w)[1]

    plt.figure(figsize=(8, 3))
    plt.rcParams.update({'font.size': 16})
    plt.subplot(121)  
    plt.fill_between(np.linspace(1, epochs, epochs), np.mean(stat_w, 0) + np.std(stat_w, 0),
                    np.mean(stat_w, 0) - np.std(stat_w, 0), alpha = 0.5, color = 'C0')
    plt.plot(np.linspace(1, epochs, epochs), np.mean(stat_w, 0), color = 'C0', linewidth = linewidth)
    plt.xlabel('Batch iterations')
    plt.ylabel(r'$\|w_f - w_b\|_F$')
    plt.grid()

    plt.subplot(122)   
    plt.fill_between(np.linspace(1, epochs, epochs), np.mean(stat_angle, 0) + np.std(stat_angle, 0),
                    np.mean(stat_angle, 0) - np.std(stat_angle, 0), alpha = 0.5, color = 'C2')
    plt.plot(np.linspace(1, epochs, epochs), np.mean(stat_angle, 0), color = 'C2', linewidth = linewidth)
    plt.xlabel('Batch iterations')
    plt.ylabel('Angle 'r'$(w_f, w_b)$')
    plt.grid()
    plt.tight_layout()
    plt.subplots_adjust(wspace = 0.5)

    plt.show()



if __name__ == '__main__':

    N_trials = len(os.listdir(os.getcwd())) - 1

    BASE_PATH = os.getcwd() +'/Trial-1/results' 
    infile = open(BASE_PATH,'rb')
    results = pickle.load(infile)
    infile.close()
    epochs = len(results['weight_tab']['dist_weight'])

    stat_w = np.zeros((N_trials, epochs))
    stat_angle = np.zeros((N_trials, epochs))


    for i in range(N_trials):
        BASE_PATH = os.getcwd() +'/Trial-' + str(i + 1) + '/results' 
        infile = open(BASE_PATH,'rb')
        results = pickle.load(infile)
        infile.close()
        #print(torch.stack(results['weight_tab']['dist_weight']))
        w_temp = torch.stack(results['weight_tab']['dist_weight']).detach().cpu().numpy()
        stat_w[i, :] = w_temp
        #print(results['weight_tab']['angle_weight'])
        angle_temp = results['weight_tab']['angle_weight']
        stat_angle[i, :] = angle_temp
        del results, w_temp, angle_temp


    print('Final weight distance: mean {:.2f}, std {:.2f}'.format(np.mean(stat_w[:, -1]), np.std(stat_w[:, -1])))
    print('Final weight angle: mean {:.2f} deg, std {:.2f} deg'.format(np.mean(stat_angle[:, -1]), np.std(stat_angle[:, -1])))

    plot_results(stat_w, stat_angle)   

    
    
    
    
    
    
