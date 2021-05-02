import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

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
    if tracking is not None:
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

def plot_potent_and_null(partitioned_samples, partitioned_null_samples, grid_1, grid_2):
    fig, axs = plt.subplots(10, 20, figsize = (20,10))
    tick_spacing = 1

    #POTENT FIRING
    for i, ax1 in enumerate(axs):
        for j, ax in enumerate(ax1):
            ax.set_aspect(1)
            
            if j<10:
                ax.scatter(partitioned_samples[i*10+j][0], partitioned_samples[i*10+j][1], marker = '.', s = 1, color = 'k')
                ax.set_xlim(grid_1[j], grid_1[j+1])
                ax.set_ylim(grid_2[i+1], grid_2[i])
                ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
                ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
                if j != 9:
                    ax.spines['right'].set_visible(False) 
            
            else:
                ax.scatter(partitioned_null_samples[i*10+j-10][0], partitioned_null_samples[i*10+j-10][1], marker = '.', s = 1, color = 'k')
                ax.set_xlim(-2.5, 2.5)
                ax.set_ylim(-2.5, 2.5)
                ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
                ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
                ax.yaxis.tick_right()
                if j != 10:
                    ax.spines['left'].set_visible(False) 
            
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=14)
            

            if j != 0 and j != 19:
                ax.set_yticks([])
            if i != 9:
                ax.set_xticks([]) 

    axs[9][5].set_xlabel(r'Potent Component $y_1$', fontsize = 24)
    axs[9][15].set_xlabel(r'Null Component $y_1$', fontsize = 24)
    axs[5][0].set_ylabel(r'Potent Component $y_2$', fontsize = 24)
    axs[5][19].yaxis.set_label_position("right")
    axs[5][19].set_ylabel(r'Null Component $y_2$', fontsize = 24)
    
    plt.show()

def plot_gaussian_contours(full_mean, full_covariance, i, j, standard_deviations = 2):

    if any(np.isnan(full_mean)):
        return [],[]
    
    #form marginal covariance
    else:
        covariance = np.array([[full_covariance[i,i], full_covariance[i,j]], 
                               [full_covariance[j,i], full_covariance[j,j]]])

        mean = np.array([full_mean[i], 
                         full_mean[j]])

        #plot a circle
        theta = np.linspace(0, 2*np.pi, 100)
        x1 = np.cos(theta)
        x2 = np.sin(theta)
        xs = np.vstack((x1,x2))

        val, vec = np.linalg.eig(covariance)
        sdxs = (standard_deviations*np.sqrt(abs(val))*xs.T).T

        theta = np.arctan(vec[1,0]/vec[0,0])

        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        sdxs = rotation_matrix@sdxs
        sdxs = (mean + sdxs.T).T
        
        return sdxs[0,:], sdxs[1,:]


def plot_all_conditional(test_case_samples, mu_n_hats, sigma_n_hats, partitioned_samples, null_direction_1, null_direction_2, contours = True):
    
    fig, axs = plt.subplots(10, 10, figsize = (11,11))
    
    for i, ax1 in enumerate(axs):
        for j, ax in enumerate(ax1):
            
            ax.scatter(test_case_samples[i*10+j][null_direction_1], test_case_samples[i*10+j][null_direction_2], marker = '.', s = 1)
            
            if contours:
                x, y = plot_gaussian_contours(mu_n_hats[i*10+j], sigma_n_hats[i*10+j], null_direction_1, null_direction_2)
                ax.plot(x,y, color = 'w', linewidth = 5)
                ax.plot(x,y, color = 'r')
            
            else:
                ax.scatter(partitioned_samples[i*10+j][null_direction_1+2], partitioned_samples[i*10+j][null_direction_2+2], marker = '.', s = 1, color = 'k')
            
            
            ax.set_aspect(1)
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.5, 2.5)
            ax.set_xticks([]) 
            ax.set_yticks([])
    plt.show()