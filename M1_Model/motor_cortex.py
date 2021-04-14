import numpy as np
import pandas as pd
import scipy as sp

from scipy import linalg
import matplotlib.pyplot as plt
from numpy.linalg import inv

from utils.plotting_functions import plot_movement, plot_two_by_four
import global_variables


def M1_dynamics(x, t, u, K, execution, linearised, standard_form):
    TAU, H_BAR, W = global_variables.TAU, global_variables.H_BAR, global_variables.W
    if execution:
        h_t = global_variables.SCALING_FACTOR * (np.exp(-(t/global_variables.TAU_DECAY)) - np.exp(-(t/global_variables.TAU_RISE)))
        dxdt = (-x + W @ (np.maximum(x, 0)) + H_BAR + h_t)/TAU
    elif linearised:
        dxdt = (-x + W @ x + H_BAR + u + K @ x )/TAU
    elif standard_form:
        dxdt = A @ x + B @ u + H_BAR
    else:
        dxdt = (-x + W @ (np.maximum(x, 0)) + H_BAR + u + K @ (np.maximum(x, 0)))/TAU
    return dxdt

def solve_M1_dynamics(x0, u = None, K = None, execution = True, linearised = False, standard_form = False):
    num_steps, t = global_variables.N_dynamics, global_variables.t_dynamics
    x = np.zeros((global_variables.NUM_NEURONS, num_steps))
    x[:, 0] = x0
    
    # Discrete Euler solver
    for step in range(num_steps - 1):
        x[:, step + 1] = x[:, step] + M1_dynamics(x[:, step], t[step], u, K, execution, linearised, standard_form) * global_variables.dt
    return x


def arm_model(phi_1, phi_2, B, torque):

    def M_func(phi):
        a_1, a_2, a_3 = global_variables.a_1, global_variables.a_2, global_variables.a_3
        M = np.array([[a_1 + 2.*a_2 * np.cos(phi[1]), a_3 + a_2*np.cos(phi[1])],[a_3 + a_2*np.cos(phi[1]), a_3]])
        return np.linalg.inv(M)
    
    def C_func(phi, phi_dot):
        C = global_variables.a_2 * np.sin(phi[1]) * np.array([-phi_dot[1] * (2.*phi_dot[0] + phi_dot[1]), phi_dot[0]**2])     
        return C

    dphi_2 = M_func(phi_1) @ (torque - C_func(phi_1, phi_2) - (B @ phi_2))
    
    return dphi_2

def solve_arm_model(torque):
    theta_init, theta_dot_init, dt, num_steps, B = global_variables.THETA_INIT, global_variables.DTHETA_INIT, global_variables.dt, global_variables.N_dynamics, global_variables.B_arm_model
    
    phi_1 = np.zeros((2, num_steps))
    phi_2 = np.zeros((2, num_steps))

    phi_1[:, 0] = theta_init
    phi_2[:, 0] = theta_dot_init
    
    for step in range(num_steps-1):
        
        phi_1[:, step + 1] = phi_1[:, step] + phi_2[:, step] * dt
        phi_2[:, step + 1] = phi_2[:, step] + arm_model(phi_1[:, step], phi_2[:, step], B, torque[:, step]) * dt

    return phi_1

if __name__ == "__main__":

    NUM_REACHES, NUM_NEURONS, N_dynamics = global_variables.NUM_REACHES, global_variables.NUM_NEURONS, global_variables.N_dynamics

    # For each of the 8 movements, simulate the dynamics of neural activity
    firing_rate_stars = np.empty((NUM_NEURONS, N_dynamics, NUM_REACHES))
    for i in range(NUM_REACHES):
        firing_rate_stars[:,:,i] = solve_M1_dynamics(global_variables.x_stars[:,i], execution = True)
        
    plot_two_by_four(firing_rate_stars, "Firing Rate", "Evolution of firing rates throughout movement execution for each direction")

    # For each set of neural activity, calculate the corresponding torque output
    torques = np.empty([2, N_dynamics, NUM_REACHES])
    for i in range(NUM_REACHES):
        torques[:,:,i]  = global_variables.C_movement @ np.maximum(firing_rate_stars[:, :, i], 0)
        
    plot_two_by_four(torques, 'Torque', 'Evolution of torques throughout movement execution for each direction')
    
    mvt_angles = np.empty((2, N_dynamics, NUM_REACHES))
    L_1, L_2 = global_variables.L_1, global_variables.L_2
    trajectories=[]
    for i in range(NUM_REACHES):
        mvt_angles[:,:,i] = solve_arm_model(torques[:,:,i])
        theta_1, theta_2 = mvt_angles[0,:,i], mvt_angles[1,:,i]
        trajectory = np.array([L_1*np.cos(theta_1)+L_2*np.cos(theta_1+theta_2), L_1*np.sin(theta_1) + L_2*np.sin(theta_1+theta_2)])
        trajectories.append(trajectory)

    plot_movement(trajectories)