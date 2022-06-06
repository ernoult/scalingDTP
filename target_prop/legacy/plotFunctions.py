import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

fontsize = 12
linewidth = 5


def plot_results(results):

    epochs = len(results["train_acc"])
    fig1 = plt.figure(figsize=(6, 4))
    plt.rcParams.update({"font.size": fontsize})
    plt.plot(np.linspace(1, epochs, epochs), results["train_acc"], linewidth=2.5, alpha=0.8)
    plt.plot(np.linspace(1, epochs, epochs), results["test_acc"], linewidth=2.5, alpha=0.8)
    plt.xlabel("Epochs")
    plt.ylabel("Train accuracy (%)")
    plt.grid()
    fig1.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    fig1.tight_layout()


if __name__ == "__main__":
    BASE_PATH = os.getcwd() + "/results"
    infile = open(BASE_PATH, "rb")
    results = pickle.load(infile)
    infile.close()
    if "train_acc" in results:
        print("Epoch {}:".format(len(results["train_acc"])))
        print("Final train accuracy: {}".format(results["train_acc"][-1]))
        print("Final test accuracy: {}".format(results["test_acc"][-1]))
    plot_results(results)
    plt.show()
