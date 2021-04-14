import numpy as np
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

W, H_BAR, x_stars, C_movement, spontaneous_firing_rates = load_parameters(NUM_NEURONS)