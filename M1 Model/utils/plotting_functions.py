import matplotlib.pyplot as plt

def plot_two_by_four(data, ylabel, title = None):
    fig, axs = plt.subplots(2, 4, figsize = (16,7), sharex = 'all', sharey = 'all')
    for i in range(2):
        for j in range(4):
            axs[i][j].plot(t_dynamics, data[:, :, 4 * i + j].T, linewidth = 1.5)
            axs[i][j].tick_params(axis='both', which='major', labelsize=12)
            axs[i][j].tick_params(axis='both', which='minor', labelsize=12)
            if i == 1:
                axs[i][j].set_xlabel('Time/s', fontsize = 12)
            if j == 0:
                axs[i][j].set_ylabel(ylabel, fontsize = 12)
    fig.suptitle(title, fontsize = 24)
    plt.tight_layout()
    plt.show()

def plot_movement(torques):
    mvt_angles = np.empty((2, N_dynamics, NUM_REACHES))

    fig, ax = plt.subplots(figsize = (10,10))

    for i in range(NUM_REACHES):
        angles = solve_arm_model(THETA_INIT, DTHETA_INIT, dt, N_dynamics, B_arm_model, torques[:,:,i])
        mvt_angles[:,:,i] = angles
        theta_1 = angles[0,:]
        theta_2 = angles[1,:]
        trajectories = np.array([L_1*np.cos(theta_1)+L_2*np.cos(theta_1+theta_2), L_1*np.sin(theta_1) + L_2*np.sin(theta_1+theta_2)])
        ax.plot(trajectories[0], trajectories[1], linewidth = 4)

    ax.set_xlabel(r'$x_1$', fontsize = 24)
    ax.set_ylabel(r'$x_2$', fontsize = 24)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    plt.show()