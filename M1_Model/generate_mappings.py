import numpy as np
import global_variables as gv

def generate_orthogonal_random_C():
    H = np.random.normal(0, 1, size = (200,2))
    q, r = np.linalg.qr(H)
    return q.T


def calculate_C_variance(list_of_Cs):
    C_variances = []
    for C in list_of_Cs:
        C_variances.append((np.trace(C@gv.covariance@C.T))/np.trace(gv.covariance))
    return C_variances

if __name__ == "__main__":

    first_120_Cs = []
    random_Cs = []
    IM_Cs = []

    for i in range(120):
        first_120_Cs.append(gv.full_basis[:,[i,i+1]].T)
        
    for i in range(30):
        random_Cs.append(generate_orthogonal_random_C())

    for i, first_vector in enumerate(gv.intrinsic_manifold[:, :-1].T):
        for second_vector in gv.intrinsic_manifold[:, i+1:].T:
            C = np.array([first_vector, second_vector])
            IM_Cs.append(C) 

    first_120_variances = calculate_C_variance(first_120_Cs)
    random_variances = calculate_C_variance(random_Cs)
    IM_variances = calculate_C_variance(IM_Cs)

    folder = 'M1_model/data/Test Mappings/'
    np.save(folder + 'first_120_Cs.npy', first_120_Cs)
    np.save(folder + 'first_120_variances.npy', first_120_variances)
    np.save(folder + 'random_Cs.npy', random_Cs)
    np.save(folder + 'random_variances.npy', random_variances)
    np.save(folder + 'IM_Cs.npy', IM_Cs)
    np.save(folder + 'IM_variances.npy', IM_variances)