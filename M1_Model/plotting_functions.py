import matplotlib.pyplot as plt
import numpy as np

import global_variables as gv

def plot_two_by_four(data, ylabel, title = None):
    fig, axs = plt.subplots(2, 4, figsize = (16,7), sharex = 'all', sharey = 'all')
    for i in range(2):
        for j in range(4):
            axs[i][j].plot(g.t_dynamics, data[:, :, 4 * i + j].T, linewidth = 1.5)
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

    for i in range(gv.NUM_REACHES):
        trajectory = trajectories[i]
        ax.plot(trajectory[0], trajectory[1], linewidth = 4)

    ax.set_xlabel(r'$x_1$', fontsize = 24)
    ax.set_ylabel(r'$x_2$', fontsize = 24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    plt.show()

def plot_covariance_eigen_values(eig_vals_1, eig_vals_2):
    fig, ax = plt.subplots(figsize = (10,5))
    ax.plot(eig_vals_1, label = 'Eigenvalues of Lyapunov Covariance', linewidth = 4)
    ax.plot(eig_vals_2, label = 'Eigenvalues of SDE Covariance Matrix', linewidth = 4)
    ax.set_xlabel(r'$i$', fontsize = 18)
    ax.set_ylabel(r'$\lambda_i$', fontsize = 18)
    ax.set_xlim(0, 50)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    ax.legend(fontsize = 16)
    plt.show()

def plot_energies(variances, solutions, title = None):
    
    IM_solutions, random_solutions, first_120_solutions = solutions
    IM_variances, random_variances, first_120_variances = variances
    
    fig, ax = plt.subplots(figsize = (10, 7))
    ax.scatter(np.log10(IM_variances), np.log10(IM_solutions), marker = 'x', color = 'r', label = "Columns of Intrinsic Manifold")
    ax.scatter(np.log10(random_variances), np.log10(random_solutions), marker = 'x', color = 'b', label = "Random Orthonormal Mapping")
    ax.scatter(np.log10(first_120_variances), np.log10(first_120_solutions), marker = 'x', color = 'g', label = r"Consecutive columns of $S$")
    ax.set_xlim(None, 0)
    ax.set_xlabel(r"$log_{10}$(Percentage Variance Explained)", fontsize = 18)
    ax.set_ylabel(r"$log_{10}$(Solution Energy)",  fontsize = 18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.legend(loc = 'lower left', fontsize = 16)
    ax.set_title(title, fontsize = 18)

    plt.show()

def plot_gp(GP, t, tracking, title = None):

    fig, ax  = plt.subplots(figsize = (10,7))
    ax.plot(t, GP.T, linewidth = 2)
    ax.plot(t, tracking.T, linewidth = 2)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.set_xlabel("Time/s", fontsize = 18)
    ax.set_ylabel("Velocity", fontsize = 18)
    ax.set_title(title, fontsize = 18)
    plt.show()

def plot_LQR(variances, state_costs, energies, ylim_lower, ylim_upper, title = None):
    
    IM_states, random_states, first_120_states = state_costs
    IM_energies, random_energies, first_120_energies = energies
    IM_variances, random_variances, first_120_variances = variances
    
    fig, axs = plt.subplots(1, 2, figsize = (20, 7))
    axs[0].scatter(np.log10(IM_variances), np.log10(IM_states), marker = 'x', color = 'r', label = "Columns of Intrinsic Manifold")
    axs[0].scatter(np.log10(random_variances), np.log10(random_states), marker = 'x', color = 'b', label = "Random Orthonormal Mapping")
    axs[0].scatter(np.log10(first_120_variances), np.log10(first_120_states), marker = 'x', color = 'g', label = r"Consecutive columns of $S$")
    
    axs[1].scatter(np.log10(IM_variances), np.log10(IM_energies), marker = 'x', color = 'r', label = "Columns of Intrinsic Manifold")
    axs[1].scatter(np.log10(random_variances), np.log10(random_energies), marker = 'x', color = 'b', label = "Random Orthonormal Mapping")
    axs[1].scatter(np.log10(first_120_variances), np.log10(first_120_energies), marker = 'x', color = 'g', label = r"Consecutive columns of $S$")
    
    
    for ax in axs:
        ax.set_xlim(None, 0)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        ax.legend(loc = 'best', fontsize = 16)
        
    axs[1].set_ylim(ylim_lower, ylim_upper)

    axs[0].set_xlabel(r"$log_{10}$(Percentage Variance Explained)", fontsize = 18)
    axs[1].set_xlabel(r"$log_{10}$(Percentage Variance Explained)", fontsize = 18)
    axs[0].set_ylabel(r"$log_{10}$(State Cost)",  fontsize = 18)
    axs[1].set_ylabel(r"$log_{10}$(Control Input Energy)",  fontsize = 18)
    
    fig.suptitle(title, fontsize = 24)
    plt.tight_layout()
    plt.show()