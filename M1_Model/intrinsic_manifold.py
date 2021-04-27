import numpy as np
import pandas as pd
import scipy as sp

from scipy import linalg
import matplotlib.pyplot as plt


from utils.plotting_functions import plot_covariance_eigen_values
import global_variables as gv


def generate_white_noise(seed, N):
    
    np.random.seed(seed)                                                                
    samples = np.random.normal(0., 1., int(N))/np.sqrt(gv.dt)
    return samples

def SDE(x, index, A, B, dW):

    dxdt = A @ x + B @ dW[:, index] + gv.H_BAR
    
    return dxdt

def solve_stochastic(x0, A, B):
    t = np.arange(0,10, gv.dt)
    num_steps = len(t)

    white_noise = np.empty((8, num_steps))
    for i in range(8):
        white_noise[i, :] = generate_white_noise(i, num_steps)

    x = np.zeros((gv.NUM_NEURONS, num_steps))
    x[:,0] = x0

    for step in range(num_steps-1):
        x[:, step+1] = x[:, step] + SDE(x[:, step], step, A, B, white_noise)*gv.dt
    return x

def find_95_percent_variance(basis, covariance):
    total_variance = np.trace(covariance)
    var = 0
    column = 0
    while var < 0.95:
        column += 1
        var = np.trace(basis[:, :column].T@ covariance @ basis[:, :column])/total_variance
    return column



if __name__=="__main__":

    A = (gv.W - np.identity(gv.NUM_NEURONS) + gv.K)/gv.TAU
    B = gv.U_tilde/gv.TAU

    SDE_solution = solve_stochastic(gv.spontaneous_firing_rates, A, B)

    P = np.cov(SDE_solution)
    eig_val_P, _ = np.linalg.eig(P)

    covariance = sp.linalg.solve_continuous_lyapunov(A, -B@B.T)
    svd_basis, svd_coeffs, _ = sp.linalg.svd(covariance)

    plot_covariance_eigen_values(svd_coeffs, eig_val_P)

    limit = find_95_percent_variance(svd_basis, COVARIANCE)

    intrinsic_manifold = svd_basis[:,:column]

    np.save("M1_Model/data/IntrinsicManifold", intrinsic_manifold) 
    np.save("M1_Model/data/covariance", covariance) 

 

