import numpy as np
import matplotlib.pyplot as plt

import GPy
import emukit.multi_fidelity
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from sklearn.preprocessing import StandardScaler
import time

#load the LES training data
training_data = np.genfromtxt('training_data.csv', delimiter = ',')
X = training_data[:,:3]
y = training_data[:,3]

#load the wake model data
ctstar_wake_model = np.genfromtxt('ctstar_wake_model.csv', delimiter=',')
wake_model = np.genfromtxt('wake_model_results_500.csv', delimiter=',')
#X_low = training_data[:,:3]
#y_low = ctstar_wake_model[:,2]
X_low = wake_model[:,:3]
y_low = wake_model[:,3]
ctstar_statistical_model = np.zeros(50)
ctstar_statistical_model_std = np.zeros(50)

from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays

for i in range(50):

    #create training and testing data
    train_index = list(filter(lambda x: x!= i, range(50)))
    test_index = i

    X_train = X[train_index,:]
    X_test = X[test_index,:][None,:]
    y_train = y[train_index][:,None]
    y_test = y[test_index]

    #scale the data
    scaler = StandardScaler()
    scaler.fit(X)
    X_train_stan_l = scaler.transform(X_low)
    X_train_stan_h = scaler.transform(X_train)
    X_train_mf, Y_train_mf = convert_xy_lists_to_arrays([X_train_stan_l, X_train_stan_h], [y_low[:,None], y[train_index,None] ])

    ## Construct a linear multi-fidelity model
    kernel_lofi = GPy.kern.RBF(input_dim=3,ARD=True) #+ GPy.kern.White(input_dim=3, variance=1e-5)
    kernel_hifi = GPy.kern.RBF(input_dim=3,ARD=True) #+ GPy.kern.White(input_dim=3, variance=1e-5)

    kernels = [kernel_lofi, kernel_hifi]
    lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)
    gpy_lin_mf_model = GPyLinearMultiFidelityModel(X_train_mf, Y_train_mf, lin_mf_kernel, n_fidelities=2)
    gpy_lin_mf_model.mixed_noise.Gaussian_noise.fix(1e-6)
    gpy_lin_mf_model.mixed_noise.Gaussian_noise_1.fix(1e-6)

    lin_mf_model = model = GPyMultiOutputWrapper(gpy_lin_mf_model, 2, n_optimization_restarts=10)

    ## Fit the model
    lin_mf_model.optimize()
    #print(gpy_lin_mf_model.mixed_noise.Gaussian_noise)
    #print(gpy_lin_mf_model.mixed_noise.Gaussian_noise_1)

    lin_mf_model.gpy_model.mixed_noise.Gaussian_noise.unfix()
    lin_mf_model.gpy_model.mixed_noise.Gaussian_noise_1.unfix()
    lin_mf_model.optimize()
    #print(gpy_lin_mf_model.mixed_noise.Gaussian_noise)
    #print(gpy_lin_mf_model.mixed_noise.Gaussian_noise_1)

    #create test points
    X_test_stan = scaler.transform(X_test)
    #X_test_mf = convert_x_list_to_array([X_test_stan, X_test_stan])
    X_test_l = np.concatenate([np.atleast_2d(X_test_stan), np.zeros((X_test_stan.shape[0], 1))], axis=1)
    X_test_h = np.concatenate([np.atleast_2d(X_test_stan), np.ones((X_test_stan.shape[0], 1))], axis=1)
    #X_test_l = X_test_mf[0,:][None,:]
    #X_test_h = X_test_mf[1,:][None,:]

    #make predictions
    lf_mean_mf_model, lf_var_mf_model = lin_mf_model.predict(X_test_l)
    hf_mean_mf_model, hf_var_mf_model = lin_mf_model.predict(X_test_h)
    print(kernels[0].lengthscale)
    print(kernels[0])
    print(kernels[1].lengthscale)
    print(kernels[1])
    ctstar_statistical_model[test_index] = hf_mean_mf_model
    ctstar_statistical_model_std[test_index] = np.sqrt(hf_var_mf_model)
    print(lf_mean_mf_model, ctstar_wake_model[test_index,2], hf_mean_mf_model, training_data[test_index,3])
    print(ctstar_statistical_model_std[test_index])

print(np.mean(np.abs(ctstar_statistical_model-training_data[:,3]))/0.75)
print(np.max(np.abs(ctstar_statistical_model-training_data[:,3]))/0.75)

np.savetxt('ctstar_statistical_model_500.csv', ctstar_statistical_model, delimiter=',')
np.savetxt('ctstar_statistical_model_std_500.csv', ctstar_statistical_model_std, delimiter=',')

plt.scatter(training_data[:,3], ctstar_wake_model[:,2], label='Wake model')
plt.scatter(training_data[:,3], ctstar_statistical_model, label='Statistical model')
plt.plot([0.55,0.8],[0.55,0.8], c='k')
plt.xlim([0.55,0.8])
plt.ylim([0.45,0.8])
plt.ylabel(r'$C_T^*$')
plt.xlabel(r'$C_{T,LES}^*$')
plt.savefig('multi_fidelity_results.png')