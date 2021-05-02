import numpy as np
from load_data import load_parameters
import os

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

W, H_BAR, x_stars, C_movement, spontaneous_firing_rates = load_parameters(NUM_NEURONS)

K = np.load('M1_Model/data/K.npy')
U_tilde = np.load('M1_Model/data/U_tilde.npy')
full_basis = np.load('M1_Model/data/full_basis.npy')
intrinsic_manifold = np.load('M1_Model/data/IntrinsicManifold.npy')
covariance = np.load('M1_Model/data/covariance.npy')

A = (W - np.identity(NUM_NEURONS) + K)/TAU
B = U_tilde/TAU

B_dash = np.zeros((202, 8))
B_dash[:200, :] = B

G_dash = np.zeros((202,2))
G_dash[200:202, :2] = np.identity(2)

M_dash = np.zeros((4, 2))
M_dash[:2, :2] = np.identity(2)

H_BAR_dash = np.zeros(202)
H_BAR_dash[:200] = H_BAR