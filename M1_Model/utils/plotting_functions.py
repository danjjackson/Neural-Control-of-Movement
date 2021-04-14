import matplotlib.pyplot as plt
import numpy as np

import global_variables

def plot_two_by_four(data, ylabel, title = None):
    fig, axs = plt.subplots(2, 4, figsize = (16,7), sharex = 'all', sharey = 'all')
    for i in range(2):
        for j in range(4):
            axs[i][j].plot(global_variables.t_dynamics, data[:, :, 4 * i + j].T, linewidth = 1.5)
            axs[i][j].tick_params(axis='both', which='major', labelsize=12)
            axs[i][j].tick_params(axis='both', which='minor', labelsize=12)
            if i == 1:
                axs[i][j].set_xlabel('Time/s', fontsize = 12)
            if j == 0:
                axs[i][j].set_ylabel(ylabel, fontsize = 12)
    fig.suptitle(title, fontsize = 24)
    plt.tight_layout()
    plt.show()

def plot_movement(trajectories):

    fig, ax = plt.subplots(figsize = (10,10))

    for i in range(global_variables.NUM_REACHES):
        trajectory = trajectories[i]
        ax.plot(trajectory[0], trajectory[1], linewidth = 4)

    ax.set_xlabel(r'$x_1$', fontsize = 24)
    ax.set_ylabel(r'$x_2$', fontsize = 24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    plt.show()