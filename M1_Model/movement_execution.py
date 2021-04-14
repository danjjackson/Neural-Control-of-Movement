import numpy as np
import pandas as pd
import scipy as sp

from scipy import linalg
import matplotlib.pyplot as plt
from numpy.linalg import inv

from utils.plotting_functions import plot_movement, plot_two_by_four
import global_variables as gv


def M1_dynamics(x, t, u, K, execution, linearised, standard_form):
    TAU, H_BAR, W = gv.TAU, gv.H_BAR, gv.W
    if execution:
        h_t = gv.SCALING_FACTOR * (np.exp(-(t/gv.TAU_DECAY)) - np.exp(-(t/gv.TAU_RISE)))
        dxdt = (-x + W @ (np.maximum(x, 0)) + H_BAR + h_t)/TAU
    elif linearised:
        dxdt = (-x + W @ x + H_BAR + u + K @ x )/TAU
    elif standard_form:
        dxdt = A @ x + B @ u + H_BAR
    else:
        dxdt = (-x + W @ (np.maximum(x, 0)) + H_BAR + u + K @ (np.maximum(x, 0)))/TAU
    return dxdt

def solve_M1_dynamics(x0, u = None, K = None, execution = True, linearised = False, standard_form = False):
    num_steps, t = gv.N_dynamics, gv.t_dynamics
    x = np.zeros((gv.NUM_NEURONS, num_steps))
    x[:, 0] = x0
    
    # Discrete Euler solver
    for step in range(num_steps - 1):
        x[:, step + 1] = x[:, step] + M1_dynamics(x[:, step], t[step], u, K, execution, linearised, standard_form) * gv.dt
    return x


def arm_model(phi_1, phi_2, B, torque):

    def M_func(phi):
        a_1, a_2, a_3 = gv.a_1, gv.a_2, gv.a_3
        M = np.array([[a_1 + 2.*a_2 * np.cos(phi[1]), a_3 + a_2*np.cos(phi[1])],[a_3 + a_2*np.cos(phi[1]), a_3]])
        return np.linalg.inv(M)
    
    def C_func(phi, phi_dot):
        C = gv.a_2 * np.sin(phi[1]) * np.array([-phi_dot[1] * (2.*phi_dot[0] + phi_dot[1]), phi_dot[0]**2])     
        return C

    dphi_2 = M_func(phi_1) @ (torque - C_func(phi_1, phi_2) - (B @ phi_2))
    
    return dphi_2

def solve_arm_model(torque):
    theta_init, theta_dot_init, dt, num_steps, B = gv.THETA_INIT, gv.DTHETA_INIT, gv.dt, gv.N_dynamics, gv.B_arm_model
    
    phi_1 = np.zeros((2, num_steps))
    phi_2 = np.zeros((2, num_steps))

    phi_1[:, 0] = theta_init
    phi_2[:, 0] = theta_dot_init
    
    for step in range(num_steps-1):
        
        phi_1[:, step + 1] = phi_1[:, step] + phi_2[:, step] * dt
        phi_2[:, step + 1] = phi_2[:, step] + arm_model(phi_1[:, step], phi_2[:, step], B, torque[:, step]) * dt

    return phi_1

def run_movement_execution(x_stars):

    NUM_REACHES, NUM_NEURONS, N_dynamics = gv.NUM_REACHES, gv.NUM_NEURONS, gv.N_dynamics

    # For each of the 8 movements, simulate the dynamics of neural activity
    firing_rate_stars = np.empty((NUM_NEURONS, N_dynamics, NUM_REACHES))
    for i in range(NUM_REACHES):
        firing_rate_stars[:,:,i] = solve_M1_dynamics(x_stars[:,i], execution = True)
        
    plot_two_by_four(firing_rate_stars, "Firing Rate", "Evolution of firing rates throughout movement execution for each direction")

    # For each set of neural activity, calculate the corresponding torque output
    torques = np.empty([2, N_dynamics, NUM_REACHES])
    for i in range(NUM_REACHES):
        torques[:,:,i]  = gv.C_movement @ np.maximum(firing_rate_stars[:, :, i], 0)
        
    plot_two_by_four(torques, 'Torque', 'Evolution of torques throughout movement execution for each direction')
    
    mvt_angles = np.empty((2, N_dynamics, NUM_REACHES))
    L_1, L_2 = gv.L_1, gv.L_2
    trajectories=[]
    for i in range(NUM_REACHES):
        mvt_angles[:,:,i] = solve_arm_model(torques[:,:,i])
        theta_1, theta_2 = mvt_angles[0,:,i], mvt_angles[1,:,i]
        trajectory = np.array([L_1*np.cos(theta_1)+L_2*np.cos(theta_1+theta_2), L_1*np.sin(theta_1) + L_2*np.sin(theta_1+theta_2)])
        trajectories.append(trajectory)

    plot_movement(trajectories)


if __name__ == "__main__":

    run_movement_execution(gv.x_stars)

    