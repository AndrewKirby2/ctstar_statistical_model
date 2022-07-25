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

    lin_mf_model = model = GPyMultiOutputWrapper(gpy_lin_mf_model, 2, n_optimization_restarts=10)

    ## Fit the model
    lin_mf_model.optimize()
    lin_mf_model.gpy_model.mixed_noise.Gaussian_noise.unfix()
    lin_mf_model.gpy_model.mixed_noise.Gaussian_noise_1.unfix()
    lin_mf_model.optimize()

    #save gp model
    with open('model.pkl', 'wb') as file:
        pickle.dump(lin_mf_model, file)

else:

    #load gp model
    with open('model.pkl', 'rb') as file:
        lin_mf_model = pickle.load(file)
