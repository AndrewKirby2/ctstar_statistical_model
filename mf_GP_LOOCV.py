"""Perform LOOCV for mf GP models
Vary the number of low fidelity
training points (n_low)
"""

import numpy as np
import matplotlib.pyplot as plt
import GPy
import emukit.multi_fidelity
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models.non_linear_multi_fidelity_model import make_non_linear_kernels, NonLinearMultiFidelityModel
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays
from sklearn.preprocessing import StandardScaler
import time

#load the LES training data
training_data = np.genfromtxt('data/LES_training_data.csv', delimiter = ',')
#remove header
training_data = np.delete(training_data, 0, 0)
training_data = np.delete(training_data, 0, 1)
X = training_data[:,:3]
y = training_data[:,3]

#load the wake model data (for comparison)
ctstar_wake_model = np.genfromtxt('data/ctstar_wake_model.csv', delimiter=',')
#remove header
ctstar_wake_model = np.delete(ctstar_wake_model, 0, 0)
ctstar_wake_model = np.delete(ctstar_wake_model, 0, 1)

#arrays to store results
ctstar_statistical_model = np.zeros((50,3))
ctstar_statistical_model_std = np.zeros((50,3))

n_low=[250,500,1000]

#loop over values of n_low
for j in range(3):
    #load low fidelity observations
    wake_model = np.genfromtxt(f'data/wake_model_maximin_{n_low[j]}.csv', delimiter=',')
    #remove header
    wake_model = np.delete(wake_model, 0, 0)
    wake_model = np.delete(wake_model, 0, 1)
    X_low = wake_model[:,:3]
    y_low = wake_model[:,3]

    train_time = np.zeros(50)
    predict_time = np.zeros(50)

    #perform LOOCV
    for i in range(50):
        print(i)
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

        # Construct a non-linear multi-fidelity model
        base_kernel = GPy.kern.RBF

        kernels = make_non_linear_kernels(base_kernel, 2, X_train_mf.shape[1] - 1, ARD=True)
        nonlin_mf_model = NonLinearMultiFidelityModel(X_train_mf, Y_train_mf, n_fidelities=2, kernels=kernels, 
                                                verbose=True, optimization_restarts=5)
        for m in nonlin_mf_model.models:
            m.Gaussian_noise.variance.fix(1e-6)

        # Fit the model
        tic = time.time()
        nonlin_mf_model.optimize()
        for m in nonlin_mf_model.models:
            m.Gaussian_noise.variance.unfix()
        nonlin_mf_model.optimize()
        toc = time.time()
        train_time[i] = toc-tic
        print(train_time[i])

        #create test points
        X_test_stan = scaler.transform(X_test)
        X_test_l = np.concatenate([np.atleast_2d(X_test_stan), np.zeros((X_test_stan.shape[0], 1))], axis=1)
        X_test_h = np.concatenate([np.atleast_2d(X_test_stan), np.ones((X_test_stan.shape[0], 1))], axis=1)

        #make predictions
        lf_mean_mf_model, lf_var_mf_model = nonlin_mf_model.predict(X_test_l)
        tic = time.time()
        hf_mean_mf_model, hf_var_mf_model = nonlin_mf_model.predict(X_test_h)
        toc = time.time()
        predict_time[i] = toc-tic
        ctstar_statistical_model[test_index,j] = hf_mean_mf_model
        ctstar_statistical_model_std[test_index,j] = np.sqrt(hf_var_mf_model)


#print results
for j in range(3):
    print('n_low = ',n_low[j])
    print('Statistical model results')
    print('MAE = ',np.mean(100*np.abs(ctstar_statistical_model[:,j]-training_data[:,3]))/0.75,'%')
    print('MAE = ',np.mean(np.abs(ctstar_statistical_model[:,j]-training_data[:,3])))
    print('Max error = ',np.max(100*np.abs(ctstar_statistical_model[:,j]-training_data[:,3]))/0.75,'%')
    print('Analytical model results')
    print('MAE = ',np.mean(100*np.abs(0.75-training_data[:,3]))/0.75,'%')
    print('MAE = ',np.mean(np.abs(0.75-training_data[:,3])))
    print('Max error = ',np.max(100*np.abs(0.75-training_data[:,3]))/0.75,'%')
    print('Wake model results')
    print('MAE = ',np.mean(100*np.abs(ctstar_wake_model[:,2]-training_data[:,3]))/0.75,'%')
    print('MAE = ',np.mean(np.abs(ctstar_wake_model[:,2]-training_data[:,3])))
    print('Max error = ',np.max(np.abs(ctstar_wake_model[:,2]-training_data[:,3]))/0.75)
    print('---------------------------------')

#save mf ctstar predictions
np.save('data/mf_GP_ctstar_predictions.npy', ctstar_statistical_model)
