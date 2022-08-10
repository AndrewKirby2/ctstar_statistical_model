import numpy as np
import matplotlib.pyplot as plt

import GPy
import emukit.multi_fidelity
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from sklearn.preprocessing import StandardScaler
import pickle

#load the LES training data
training_data = np.genfromtxt('training_data.csv', delimiter = ',')
X_high = training_data[:,:3]
y_high = training_data[:,3]

#load the wake model data
ctstar_wake_model = np.genfromtxt('ctstar_wake_model.csv', delimiter=',')
wake_model = np.genfromtxt('wake_model_results_1000.csv', delimiter=',')
X_low = wake_model[:,:3]
y_low = wake_model[:,3]
ctstar_statistical_model = np.zeros(50)
ctstar_statistical_model_std = np.zeros(50)

from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays

fit_model = True

if fit_model:

    #scale the data
    scaler = StandardScaler()
    scaler.fit(X_high)
    X_train_stan_l = scaler.transform(X_low)
    X_train_stan_h = scaler.transform(X_high)
    X_train_mf, Y_train_mf = convert_xy_lists_to_arrays([X_train_stan_l, X_train_stan_h], [y_low[:,None], y_high[:,None] ])

    # Construct a linear multi-fidelity model
    kernel_lofi = GPy.kern.RBF(input_dim=3,ARD=True)
    kernel_hifi = GPy.kern.RBF(input_dim=3,ARD=True)

    kernels = [kernel_lofi, kernel_hifi]
    lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)
    gpy_lin_mf_model = GPyLinearMultiFidelityModel(X_train_mf, Y_train_mf, lin_mf_kernel, n_fidelities=2)
    gpy_lin_mf_model.mixed_noise.Gaussian_noise.fix(1e-6)
    gpy_lin_mf_model.mixed_noise.Gaussian_noise_1.fix(1e-6)

    lin_mf_model = model = GPyMultiOutputWrapper(gpy_lin_mf_model, 2, n_optimization_restarts=5)

    ## Fit the model
    lin_mf_model.optimize()
    lin_mf_model.gpy_model.mixed_noise.Gaussian_noise.unfix()
    lin_mf_model.gpy_model.mixed_noise.Gaussian_noise_1.unfix()
    lin_mf_model.optimize()

    #save gp model
    with open('lin_model.pkl', 'wb') as file:
        pickle.dump(lin_mf_model, file)

else:

    #load gp model
    with open('lin_model.pkl', 'rb') as file:
        lin_mf_model = pickle.load(file)
    
    print(lin_mf_model.gpy_model)
    print(lin_mf_model.gpy_model.multifidelity.rbf.lengthscale)
    print(lin_mf_model.gpy_model.multifidelity.rbf_1.lengthscale)

#create test points
n_x = 50
n_y = 50
x = np.linspace(500, 1000, n_x)
y = np.linspace(500, 1000, n_y)
xx, yy = np.meshgrid(x,y)

#loop over different values for theta
for z in np.arange(0,46,5):

    X_test = np.array([xx.flatten(),yy.flatten(),z*np.ones(n_x*n_y)]).transpose()
    scaler = StandardScaler()
    scaler.fit(X_high)
    X_test_stan = scaler.transform(X_test)
    X_test_l = np.concatenate([np.atleast_2d(X_test_stan), np.zeros((X_test_stan.shape[0], 1))], axis=1)
    X_test_h = np.concatenate([np.atleast_2d(X_test_stan), np.ones((X_test_stan.shape[0], 1))], axis=1)

    lf_mean_mf_model, lf_var_mf_model = lin_mf_model.predict(X_test_l)
    hf_mean_mf_model, hf_var_mf_model = lin_mf_model.predict(X_test_h)
    hf_mean_mf_model = np.reshape(hf_mean_mf_model.flatten(),(n_x,n_y))
    lf_mean_mf_model = np.reshape(lf_mean_mf_model.flatten(),(n_x,n_y))
    lf_var_mf_model = np.reshape(lf_var_mf_model.flatten(),(n_x,n_y))
    hf_var_mf_model = np.reshape(hf_var_mf_model.flatten(),(n_x,n_y))

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=[8,3])
    pcm_lf = ax[0].pcolormesh(xx, yy, lf_mean_mf_model, vmin=0.5, vmax=0.8)
    ax[0].set_xlabel(r'$S_x$ (m)')
    ax[0].set_ylabel(r'$S_y$ (m)')
    ax[0].set_title(r'a) $f_{low}(x)$', loc = 'left')
    cbar = fig.colorbar(pcm_lf, ax=ax[0], shrink=0.97)
    cbar.set_label(r'Posterior mean')

    pcm_hf = ax[1].pcolormesh(xx, yy, hf_mean_mf_model, vmin=0.5, vmax=0.8)
    ax[1].set_xlabel(r'$S_x$ (m)')
    ax[1].set_ylabel(r'$S_y$ (m)')
    ax[1].set_title(r'b) $f_{high}(x)$', loc = 'left')
    plt.tight_layout()
    cbar = fig.colorbar(pcm_hf, ax=ax[1], shrink=0.97)
    cbar.set_label(r'Posterior mean')
    plt.savefig(f'mf_gp_posterior/lin_posterior_mean_theta{z}.png')
    plt.close()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=[8,3])
    pcm_lf = ax[0].pcolormesh(xx, yy, lf_var_mf_model, vmin=0, vmax=1e-4)
    ax[0].set_xlabel(r'$S_x$ (m)')
    ax[0].set_ylabel(r'$S_y$ (m)')
    ax[0].set_title(r'a) $f_{low}(x)$', loc = 'left')
    cbar = fig.colorbar(pcm_lf, ax=ax[0], shrink=0.97)
    cbar.set_label(r'Posterior variance')

    pcm_hf = ax[1].pcolormesh(xx, yy, hf_var_mf_model, vmin=0, vmax=1e-4)
    ax[1].set_xlabel(r'$S_x$ (m)')
    ax[1].set_ylabel(r'$S_y$ (m)')
    ax[1].set_title(r'b) $f_{high}(x)$', loc = 'left')
    plt.tight_layout()
    cbar = fig.colorbar(pcm_hf, ax=ax[1], shrink=0.97)
    cbar.set_label(r'Posterior variance')
    plt.savefig(f'mf_gp_posterior/lin_posterior_std_theta{z}.png')
    plt.close()
