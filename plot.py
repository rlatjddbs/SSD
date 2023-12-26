from matplotlib import pyplot as plt
import numpy as np
import pdb

def plot(y, label):
    fig, ax = plt.subplots()
    x = np.arange(0.2, 1.2, 0.2)
    for i, j in enumerate(y):
        ax.plot(x, j, label=label[i])
        ax.legend()
    plt.xlabel("Target value")
    plt.ylabel("Normalized score")
    plt.title("Maze2d-Umaze")
    plt.tight_layout()
    plt.show()
    
x = np.array([[124.5, 129.7, 118.8, 132.3, 147.4], [115.9, 124.7, 121.1, 121.6, 127.7]])
label = ["Multi", "Single"]

plot(x, label)