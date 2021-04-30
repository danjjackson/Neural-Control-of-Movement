import numpy as np
import pandas as pd
import scipy as sp
from scipy import linalg
import matplotlib.pyplot as plt
from numpy.linalg import inv
#from tqdm import tqdm

from plotting_functions import plot_energies
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

def solution(C, v_star):
    
    def M(C):
        M = - C @ np.linalg.inv(A) @ B
        return M

    def N(C):
        N = - C @ np.linalg.inv(A) @ gv.H_BAR
        return N
    
    u, s, v_T = np.linalg.svd(M(C), full_matrices = False)
    
    solution =  v_T.T @ (np.diag(1/s)) @ u.T @ (v_star - N(C))
    
    return solution, np.linalg.norm(solution)

def steady_state(target):

    ss_solutions = []
    for Cs in all_Cs:
        ss_energies = []
        for C in Cs:
            ss_energies.append(solution(C, target)[1])
            
        ss_solutions.append(ss_energies)

    return ss_solutions


if __name__=="__main__":

    target_velocity = np.array([1,-1])
    steady_state_solution = steady_state(target_velocity)

    plot_energies(variances, steady_state_solution, "Steady-state solution energy plotted against percentage variance accounted for by the mapping")