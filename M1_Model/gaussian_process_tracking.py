import numpy as np
import pandas as pd
import scipy as sp
from scipy import linalg
import matplotlib.pyplot as plt
from numpy.linalg import inv
from tqdm import tqdm

from plotting_functions import plot_energies, plot_gp, plot_LQR
import global_variables as gv


mapping_folder = 'M1_Model/data/Test Mappings/'
IM_variances = np.load(mapping_folder + 'IM_variances.npy')
random_variances = np.load(mapping_folder + 'random_variances.npy')
first_120_variances = np.load(mapping_folder + 'first_120_variances.npy')
IM_Cs = np.load(mapping_folder + "IM_Cs.npy")
random_Cs = np.load(mapping_folder + 'random_Cs.npy')
first_120_Cs = np.load(mapping_folder + 'first_120_Cs.npy')
all_Cs = [IM_Cs, random_Cs, first_120_Cs]
variances = [IM_variances, random_variances, first_120_variances]

A = (gv.W - np.identity(gv.NUM_NEURONS) + gv.K)/gv.TAU
B = gv.U_tilde/gv.TAU

B_dash = np.zeros((202, 8))
B_dash[:200, :] = B

G_dash = np.zeros((202,2))
G_dash[200:202, :2] = np.identity(2)

M_dash = np.zeros((4, 2))
M_dash[:2, :2] = np.identity(2)

H_BAR_dash = np.zeros(202)
H_BAR_dash[:200] = gv.H_BAR


length_scale = 0.15

def generate_GP(time, variance = 40):
    length_scale = 0.15
    t = np.arange(0, 2 * time, gv.dt)
    N = len(t)
    DFT_N = int(N/2)+1
    GP = np.empty((2, DFT_N))

    for j in range(2):
        fourier_vector = np.fft.rfft(np.random.normal(0, variance, N))
        omega  = np.linspace(0, 2*np.pi/gv.dt, len(fourier_vector))
        fourier_covariance = np.exp(-(omega*length_scale/2)**2/2)
        GP[j,:] = np.fft.ifft(fourier_covariance*fourier_vector)
        
    return GP, t[:DFT_N], DFT_N

def LQI(A, C, lmbda_Q = 1, lmbda_R = 1):

    #Extend matrices
    A_dash = np.zeros((202, 202))
    A_dash[:200, :200] = A
    A_dash[200:202, :200] = - C
    
    C_dash = np.zeros((4,202))
    C_dash[:2, :200] = - C
    C_dash[2:, 200:202] = np.identity(2)
    
    R = lmbda_R * np.identity(8)
    Q = lmbda_Q * np.identity(4)
    
    #Solve Riccati Equaiton
    X_ss = sp.linalg.solve_continuous_are(A_dash, B_dash, C_dash.T @ Q @ C_dash, R)
    
    #OPTIMAL CONTROLLER
    K_x = np.linalg.inv(R) @ B_dash.T @ X_ss
    K_v = np.linalg.inv(R) @ B_dash.T @ np.linalg.inv(X_ss @ B_dash @ np.linalg.inv(R) @ B_dash.T - A_dash.T)@(C_dash.T @ Q @ M_dash + X_ss@G_dash)
    
    #Closed loop dynamics
    A_BAR = A_dash - B_dash @ K_x
    B_BAR = G_dash - B_dash @ K_v
    
    return A_BAR, B_BAR, K_x, K_v, C_dash

def tracking(A, B, x, v, H_BAR_dash):
    dxdt = A @ x + B @ v + H_BAR_dash
    return dxdt

def solve_tracking(x0, num_steps, A, B, v, K_x, K_v, C_dash, H_BAR_dash):
    x = np.zeros((202, num_steps))
    x[:200,0] = x0

    for step in range(num_steps-1):
        x[:, step+1] = x[:, step] + tracking(A, B, x[:, step], v[:, step], H_BAR_dash) * gv.dt

    #This is an 8 * 4000 array of all 8 control inputs at every one of the 4000 time steps 
    u = - K_x @ x - K_v @ v
    
    #Find the squared norm of each set of 8 inputs, then sum across all 4000 time steps
    uTu = 0
    for i in range(num_steps):
        uTu += u[:,i].T@u[:,i]

    return x, uTu

def state_LQR(A, B, C, x_star, lmbda_R = 1, lmbda_Q = 1):
    
    R = lmbda_R * np.identity(8)
    Q = lmbda_Q * np.identity(200)

    X = sp.linalg.solve_continuous_are(A, B, Q, R)
    J = x_star @ X @ x_star
    
    K = - (np.linalg.inv(R) @ B.T @ X)
    P = sp.linalg.solve_continuous_lyapunov((A + B @ K).T, -K.T @ K) 

    input_energy = x_star @ P @ x_star
    uRu =  input_energy * lmbda_R
    xQx = J - uRu
    state_cost = xQx/lmbda_Q
    
    return input_energy, state_cost

def solve_state_LQR(target_velocity):
    all_energies = []
    all_state_costs = []

    for Cs in all_Cs:

        uTus = [] 
        xTxs = [] 

        for C in tqdm(Cs):
            required_x = np.linalg.pinv(C) @ target_velocity
            uTu, xTx = state_LQR(A, B, C, required_x)
            uTus.append(uTu), xTxs.append(xTx)
            
        all_energies.append(uTus), all_state_costs.append(xTxs)
    
    return all_energies, all_state_costs


def bisect(A, lmbda_low, lmbda_high, x, C, target):
    
    lmbda_mid = (lmbda_low + lmbda_high) / 2.0

    print('lambda = {}'.format(lmbda_mid))

    uTu, xTx = state_LQR(A, B, C, x, lmbda_R = lmbda_mid, lmbda_Q = 1)
    print('uTu = {}'.format(uTu))

    if uTu < target:
        lmbda_high =  lmbda_mid
    else:
        lmbda_low = lmbda_mid

    return lmbda_low, lmbda_high, lmbda_mid, uTu


def LQR_bisection(A, lmbda_high, lmbda_low, target, v, B, C):
    
    x = np.linalg.pinv(C) @ v

    uTu, xTx = state_LQR(A, B, C, x, lmbda_R = lmbda_high, lmbda_Q = 1)
    
    print('Initial lambda is {}'.format(lmbda_high))›
    print('With corresponding starting energy {}'.format(uTu))

    while uTu > target:
        lmbda_high = lmbda_high * 2
        print('Increasing lmbda high - now lmbda = {}'.format(lmbda_high))
        uTu, xTx = state_LQR(A, B, C, x, lmbda_R = lmbda_high, lmbda_Q = 1)
        print(uTu)

    lmbda_final = lmbda_high

    while abs(uTu - target) > 0.000001:

        lmbda_low, lmbda_high, lmbda_mid, uTu = bisect(A, lmbda_low, lmbda_high, x, C, target, bisect_energy)
        lmbda_final = lmbda_mid

    uTu, xTx = state_LQR(A, B, C, x, lmbda_R = lmbda_final, lmbda_Q = 1)

    return uTu, xTx

def solve_state_LQR_bisected(target_velocity)

    constant_energies = []
    variable_state_costs = []

    for Cs in all_Cs:

        target_energy = 10**(-3)

        lmbda_high = 10
        lmbda_low = 0

        energies = [] 
        state_costs = []

        for C in tqdm(Cs):
            energy, state_cost = LQR_bisection(A, lmbda_high, lmbda_low, target_energy, target_velocity, B, C, bisect_energy = True)
            energies.append(energy), state_costs.append(state_cost)
        
        constant_energies.append(energies), variable_state_costs.append(state_costs)
    return constant_energies, variable_state_costs



if __name__ == "__main__":

    GP, t_tracking, N_tracking = generate_GP(1)
    #plot_gp(GP, t_tracking, tracking = None)

    test_C = gv.intrinsic_manifold[:, [0, 1]].T
    A_BAR, B_BAR, K_x, K_v, C_dash, = LQI(A, test_C)
    x_tracking, tracking_energy = solve_tracking(gv.spontaneous_firing_rates, N_tracking, A_BAR, B_BAR, GP, K_x, K_v, C_dash, H_BAR_dash)
    GP_tracking = test_C @ x_tracking[:200, :]

    #plot_gp(GP, t_tracking, GP_tracking, "Tracking a 2d Gaussian Process using LQI control")

    target_velocity = np.array([1,-1])
    all_energies, all_state_costs = solve_state_LQR(target_velocity)

    plot_LQR(variances, all_state_costs, all_energies, -3, None, "State Costs and Control Input Energy vs Percentage Variance accounted for")

    constant_energies, variable_state_costs = solve_state_LQR_bisected(target_velocity)
    plot_LQR(variances, variable_state_costs, constant_energies, -3, None, "State Costs and Control Input Energy vs Percentage Variance accounted for")