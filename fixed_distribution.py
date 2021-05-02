import numpy as np
import scipy as sp
from scipy import linalg
import matplotlib.pyplot as plt
from numpy.linalg import inv
from tqdm import tqdm
from sklearn.utils import shuffle
import matplotlib.ticker as ticker

import global_variables as gv

from gaussian_process_tracking import generate_GP, LQI, tracking, solve_tracking
from plotting_functions import plot_gp, plot_potent_and_null, plot_gaussian_contours, plot_all_conditional

def track_sections(gaussian_process, parts, length, A_BAR, B_BAR, C, Q, K_x, K_v):

    sectioned_GP = []

    for i in range(parts):
        section = gaussian_process[:, i*length:(i+1)*length+1]
        sectioned_GP.append(section)

    x_activity_sections = []
    y_activity_sections = []
    v_tracking_sections = []

    for i, section in tqdm(enumerate(sectioned_GP)):
        x_ext, _ = solve_tracking(length+1, A_BAR, B_BAR, section, K_x, K_v)
        
        x = x_ext[:200,:]
        y = Q.T @ x
        velocity = C @ x
        
        if i == 0:
            x_activity_sections.append(x)
            y_activity_sections.append(y)
            v_tracking_sections.append(velocity)
            
        else:
            x_activity_sections.append(np.delete(x, 0, 1))
            y_activity_sections.append(np.delete(y, 0, 1))
            v_tracking_sections.append(np.delete(velocity, 0, 1))
            
    x = np.hstack(x_activity_sections)
    y = np.hstack(y_activity_sections)
    v = np.hstack(v_tracking_sections)

    return x, y, v

def partition_data(y):
    
    y_potent_1 = y[0,:]
    y_potent_2 = y[1,:]
    y_null_1 = y[2, :]
    y_null_2 = y[3, :]
    y_null_3 = y[4, :]
    y_null_4 = y[5, :]
    
    x1_upper = np.ceil(max(y_potent_1))
    x1_lower = -np.ceil(abs(min(y_potent_1)))
    x2_upper = np.ceil((max(y_potent_2)))
    x2_lower = -np.ceil(abs(min(y_potent_2)))
    
    grid_1 = np.linspace(x1_lower, x1_upper, 11)
    grid_2 = np.linspace(x2_upper, x2_lower, 11)

    grid_centres_1 = [np.round((grid_1[i]+grid_1[i+1])/2,2) for i in range(len(grid_1)-1)]
    grid_centres_2 = [np.round((grid_2[i]+grid_2[i+1])/2,2) for i in range(len(grid_2)-1)]

    list_of_grid_squares = []

    for i in tqdm(range(10)):
        for j in range(10):

            grid_square_ij = []

            grid_square_ij_y_potent_1 = []
            grid_square_ij_y_potent_2 = []
            grid_square_ij_y_null_1 = []
            grid_square_ij_y_null_2 = []
            grid_square_ij_y_null_3 = []
            grid_square_ij_y_null_4 = []

            for d in range(np.shape(y)[1]):
                if grid_1[j] <= y_potent_1[d] and y_potent_1[d] < grid_1[j+1] and grid_2[i] > y_potent_2[d] and y_potent_2[d] >= grid_2[i+1]:
                    grid_square_ij_y_potent_1.append(y_potent_1[d])
                    grid_square_ij_y_potent_2.append(y_potent_2[d])
                    grid_square_ij_y_null_1.append(y_null_1[d])
                    grid_square_ij_y_null_2.append(y_null_2[d])
                    grid_square_ij_y_null_3.append(y_null_3[d])
                    grid_square_ij_y_null_4.append(y_null_4[d])             
                else:
                    continue

            grid_square_ij.append(grid_square_ij_y_potent_1) 
            grid_square_ij.append(grid_square_ij_y_potent_2)
            grid_square_ij.append(grid_square_ij_y_null_1)
            grid_square_ij.append(grid_square_ij_y_null_2)
            grid_square_ij.append(grid_square_ij_y_null_3)
            grid_square_ij.append(grid_square_ij_y_null_4)

            list_of_grid_squares.append(grid_square_ij)
    
    return list_of_grid_squares, grid_centres_1, grid_centres_2, grid_1, grid_2

def conditional_Gaussian_parameters(data, grid_centres_1, grid_centres_2, grid_coord): 

    yp1 = np.asarray(data[grid_coord][0])
    yp2 = np.asarray(data[grid_coord][1])
    
    yn1 = np.asarray(data[grid_coord][2])
    yn2 = np.asarray(data[grid_coord][3])
    yn3 = np.asarray(data[grid_coord][4])
    yn4 = np.asarray(data[grid_coord][5])
    
    y = np.vstack([yp1, yp2, yn1, yn2, yn3, yn4])
    N = np.shape(y)[1]
    
    mu_p = np.mean(y[:2, :], axis = 1)
    mu_n = np.mean(y[2:, :], axis = 1)
    cov = np.cov(y)

    sigma_pp = cov[:2, :2]
    sigma_nn = cov[2:, 2:]
    sigma_pn = cov[:2, 2:]
    
    y_p = np.array([grid_centres_1[grid_coord%10], grid_centres_2[int((grid_coord-grid_coord%10)/10)]])

    mu_n_hat = mu_n + sigma_pn.T@np.linalg.inv(sigma_pp) @ (y_p - mu_p)
    sigma_n_hat = sigma_nn - sigma_pn.T@np.linalg.inv(sigma_pp)@sigma_pn
    
    return mu_n_hat, sigma_n_hat, N

def generate_samples(partitioned_samples, grid_centres_1, grid_centres_2):

    FD_null_samples = []  # List for samples generated according to the Fixed Distribution Hypothesis
    mu_n_hats = []
    sigma_n_hats = []

    # For each grid square
    for i in tqdm(range(10)):
        for j in range(10):
            
            # Calculate the parameters of the conditional gaussian distribution and store them in a list
            mu_n_hat, sigma_n_hat, N_data_points = conditional_Gaussian_parameters(partitioned_samples, grid_centres_1, grid_centres_2, i*10+j)
            mu_n_hats.append(mu_n_hat)
            sigma_n_hats.append(sigma_n_hat)
            
            # Draw samples from a multivariate gaussian distribution parameterised by the calculated mean and covariance
            if any(np.isnan(mu_n_hat)):
                FD_null_samples.append([[],[],[],[]])
            else:
                samples = np.random.multivariate_normal(mu_n_hat, sigma_n_hat, N_data_points)
                FD_null_samples.append([*samples.T])

    return FD_null_samples, mu_n_hats, sigma_n_hats

def plot_single_conditional(test_case_samples, partitioned_samples, partitioned_null_samples, grid_square, null_direction_1, null_direction_2, grid_centres_1, grid_centres_2,contours = True):
    
    fig, ax = plt.subplots(figsize = (14,7))
    
    if contours:
        mu_n_hat, Sigma_n_hat, N_data_points = conditional_Gaussian_parameters(partitioned_samples, grid_centres_1, grid_centres_2, grid_square)
        gx, gy = plot_gaussian_contours(mu_n_hats[grid_square], sigma_n_hats[grid_square], null_direction_1, null_direction_2)
        ax.plot(gx,gy, color = 'w', linewidth = 5)
        ax.plot(gx,gy, color = 'r')
    
    ax.scatter(test_case_samples[grid_square][null_direction_1], test_case_samples[grid_square][null_direction_2], marker = '.', s = 10)
    ax.scatter(partitioned_null_samples[grid_square][null_direction_1], partitioned_null_samples[grid_square][null_direction_2+2], marker = '.', s = 10, color = 'k')
    
    ax.set_aspect(1)
    ax.set_xlabel('Null Component {}'.format(null_direction_1+1), fontsize = 24)
    ax.set_ylabel('Null Component {}'.format(null_direction_2+1), fontsize = 24)
    ax.set_xlim(-2.5,2.5)
    ax.set_ylim(-2.5,2.5)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=16)

    plt.show()

if __name__ == "__main__":
    long_GP, t, N = generate_GP(300)

    #DENOTE THE ORTHOGONAL COMPLEMENT OF THE CHOSEN COLUMNS OF THE INTRINSIC MANIFOLD AS C_BAR
    C = gv.intrinsic_manifold[:, [0,1]].T
    A_BAR, B_BAR, K_x, K_v = LQI(C)

    u, sigma, vT = sp.linalg.svd(C)
    Q = vT.T
    C_BAR = Q[:, 2:]

    parts = 30
    length = int((N-1)/parts)

    x, y, v = track_sections(long_GP, parts, length, A_BAR, B_BAR, C, Q, K_x, K_v)

    #plot_gp(long_GP[:, 30000:40000], t[30000:40000], v[:, 30000:40000])

    partitioned_samples, grid_centres_1, grid_centres_2, grid_1, grid_2 = partition_data(y)
    partitioned_null_samples = [grid_square[2:] for grid_square in partitioned_samples]

    #plot_potent_and_null(partitioned_samples, partitioned_null_samples, grid_1, grid_2)

    FD_null_samples, mu_n_hats, sigma_n_hats = generate_samples(partitioned_samples, grid_centres_1, grid_centres_2)

    #plot_single_conditional(FD_null_samples, partitioned_samples, partitioned_null_samples, 45, 0, 1, grid_centres_1, grid_centres_2)
    plot_all_conditional(FD_null_samples, mu_n_hats, sigma_n_hats, partitioned_samples, 0,1)