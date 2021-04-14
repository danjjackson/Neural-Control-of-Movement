import numpy as np
import pandas as pd
import scipy as sp
from scipy import linalg
import matplotlib.pyplot as plt
from numpy.linalg import inv

from utils.plotting_functions import plot_movement, plot_two_by_four
from utils.load_data import load_parameters


### GLOBAL VARIABLES ###

NUM_REACHES = 8
NUM_NEURONS = 200

###### EULER SOLVER #####
dt = 0.0005
t_dynamics = np.arange(0, 1, dt)
N_dynamics = len(t_dynamics)

###### M1 DYNAMICS ######
TAU = 0.15
TAU_RISE = 0.5
TAU_DECAY = 0.05

h_t_MAX = 5
h_t_MAX_TIME = (np.log(TAU_RISE/TAU_DECAY))/((1/TAU_DECAY)-(1/TAU_RISE))
SCALING_FACTOR = h_t_MAX/(np.exp(-h_t_MAX_TIME/TAU_DECAY)-np.exp(-h_t_MAX_TIME/TAU_RISE))  #h_t is an alpha shaped input bump with the scaling factor set such that it has a maximum value of 5

###### ARM MECHANICS AND HAND TRAJECTORIES ######
THETA_INIT = np.array([10*np.pi/180, 143.54*np.pi/180])
DTHETA_INIT = np.array([0, 0])
L_1 = 0.3
L_2 = 0.33
I_1 = 0.025
I_2 = 0.045
M_1 = 1.4
M_2 = 1.0
D_2 = 0.16

a_1 = I_1 + I_2 + M_2 * (L_1**2)
a_2 = M_2 * L_1 * D_2
a_3 = I_2

B_arm_model = np.array([[0.05, 0.025], [0.025, 0.05]])

##### Full Circuit Model ######
LAMBDA = 0.1


def M1_dynamics(x, t, u, K, execution, linearised, standard_form):
    if execution:
        h_t = SCALING_FACTOR * (np.exp(-(t/TAU_DECAY)) - np.exp(-(t/TAU_RISE)))
        dxdt = (-x + W @ (np.maximum(x, 0)) + H_BAR + h_t)/TAU
    elif linearised:
        dxdt = (-x + W @ x + H_BAR + u + K @ x )/TAU
    elif standard_form:
        dxdt = A @ x + B @ u + H_BAR
    else:
        dxdt = (-x + W @ (np.maximum(x, 0)) + H_BAR + u + K @ (np.maximum(x, 0)))/TAU
    return dxdt

def solve_M1_dynamics(x0, dt, num_steps, t, u = None, K = None, execution = True, linearised = False, standard_form = False):
    x = np.zeros((NUM_NEURONS, num_steps))
    x[:, 0] = x0
    
    # Discrete Euler solver
    for step in range(num_steps - 1):
        x[:, step + 1] = x[:, step] + M1_dynamics(x[:, step], t[step], u, K, execution, linearised, standard_form) * dt
    return x


def M_func(phi):
    M = np.array([[a_1 + 2.*a_2 * np.cos(phi[1]), a_3 + a_2*np.cos(phi[1])],[a_3 + a_2*np.cos(phi[1]), a_3]])
    return np.linalg.inv(M)
    
def C_func(phi, phi_dot):
    C = a_2 * np.sin(phi[1]) * np.array([-phi_dot[1] * (2.*phi_dot[0] + phi_dot[1]), phi_dot[0]**2])     
    return C
    

def arm_model(phi_1, phi_2, B, torque, M_func, C_func):

    dphi_2 = M_func(phi_1) @ (torque - C_func(phi_1, phi_2) - (B @ phi_2))
    
    return dphi_2

def solve_arm_model(theta_init, theta_dot_init, dt, num_steps, B, torque):
    
    phi_1 = np.zeros((2, num_steps))
    phi_2 = np.zeros((2, num_steps))

    phi_1[:, 0] = theta_init
    phi_2[:, 0] = theta_dot_init
    
    for step in range(num_steps-1):
        
        phi_1[:, step + 1] = phi_1[:, step] + phi_2[:, step] * dt
        phi_2[:, step + 1] = phi_2[:, step] + arm_model(phi_1[:, step], phi_2[:, step], B, torque[:, step]) * dt

    return phi_1

if __name__ == "__main__":

    W, spontaneous_firing_rates, H_BAR, x_stars, C_movement = load_parameters()
    
    # For each of the 8 movements, simulate the dynamics of neural activity
    firing_rate_stars = np.empty((NUM_NEURONS, N_dynamics, NUM_REACHES))
    for i in range(NUM_REACHES):
        firing_rate_stars[:,:,i] = solve_M1_dynamics(x_stars[:,i], dt, N_dynamics, t_dynamics, H_BAR, SCALING_FACTOR, execution = True)
        
    plot_two_by_four(firing_rate_stars, "Firing Rate", "Evolution of firing rates throughout movement execution for each direction")

    # For each set of neural activity, calculate the corresponding torque output
    torque_stars = np.empty([2, N_dynamics, NUM_REACHES])
    for i in range(NUM_REACHES):
        torque_stars[:,:,i]  = C_movement @ np.maximum(firing_rate_stars[:, :, i], 0)
        
    plot_two_by_four(torque_stars, 'Torque', 'Evolution of torques throughout movement execution for each direction')

    plot_movement(torque_stars)