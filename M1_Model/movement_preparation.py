import numpy as np
import pandas as pd
import scipy as sp

from scipy import linalg
import matplotlib.pyplot as plt
from numpy.linalg import inv

from utils.plotting_functions import plot_movement, plot_two_by_four
import global_variables as gv

from movement_execution import solve_M1_dynamics, run_movement_execution


def observability_gramian():
    I = np.identity(gv.NUM_NEURONS)
    W_BAR = gv.W - I
    #Solve Lyapunov Equation to find the observability gramian Q
    Q_obsv= sp.linalg.solve_continuous_lyapunov(W_BAR.T, -1*(gv.TAU * gv.C_movement.T @ gv.C_movement))
    #Solve the Algebraic Riccati Equation (ARE)
    K = -(gv.LAMBDA**-1) * sp.linalg.solve_continuous_are(W_BAR, I, Q_obsv, (gv.LAMBDA**-1)*I)
    return K


if __name__=="__main__":
    K = observability_gramian()
    
    #Calculate columns of U that will map to correct initial conditions (optimal subspace)
    U_tilde = np.empty((gv.NUM_NEURONS, gv.NUM_REACHES))
    test_x_stars = np.empty((gv.NUM_NEURONS, gv.NUM_REACHES))

    for i in range(gv.NUM_REACHES):
        U_tilde[:,i] =  gv.x_stars[:,i] - (gv.W + K) @ np.maximum(gv.x_stars[:,i],0) - gv.H_BAR
        prep_firing_rates = solve_M1_dynamics(gv.spontaneous_firing_rates, U_tilde[:,i], K, execution = False)
        test_x_stars[:,i] = prep_firing_rates[:, -1]

    run_movement_execution(test_x_stars)
