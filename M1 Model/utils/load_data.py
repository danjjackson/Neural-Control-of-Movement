import pandas as pd
import numpy as np

def load_parameters()
    # Load all the model parameter files
    model_folder = 'Network Model Parameters/'
    W = pd.read_csv(model_folder + 'W_matrix.txt', header = None, delimiter = '\t').dropna(axis=1,how='all').values
    spontaneous_firing_rates = np.reshape(pd.read_csv(model_folder + 'spontaneous_firing.txt', header = None, delimiter = '\t').dropna(axis=1, how='all').values, 200)
    H_BAR = np.reshape(spontaneous_firing_rates - W @ np.maximum(spontaneous_firing_rates, 0), 200)
    x_stars = pd.read_csv(model_folder + 'x_stars.txt', header = None, delimiter = '\t').dropna(axis=1, how='all').values + np.reshape(spontaneous_firing_rates, (200, 1))

    C_movement = np.zeros((2, NUM_NEURONS))
    C_movement[:, :160] = pd.read_csv(model_folder + 'C_matrix.txt', header = None, delimiter = '\t').dropna(axis=1, how='all').values
    return W, spontaneous_firing_rates, H_BAR, x_stars, C_movement