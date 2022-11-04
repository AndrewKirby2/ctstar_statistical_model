import numpy as np
import matplotlib.pyplot as plt

import GPy
import emukit.multi_fidelity
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models.non_linear_multi_fidelity_model import make_non_linear_kernels, NonLinearMultiFidelityModel
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays
from sklearn.preprocessing import StandardScaler
import string

#load the LES training data
training_data = np.genfromtxt('training_data.csv', delimiter = ',')
X_high = training_data[:,:3]
y_high = training_data[:,3]

#load the wake model data
#ctstar_wake_model = np.genfromtxt('ctstar_wake_model.csv', delimiter=',')
wake_model = np.genfromtxt('data/wake_model_results_500.csv', delimiter=',')
X_low = wake_model[:,:3]
y_low = wake_model[:,3]
ctstar_statistical_model = np.zeros(50)
ctstar_statistical_model_std = np.zeros(50)

#scale the data
scaler = StandardScaler()
scaler.fit(X_high)
X_train_stan_l = scaler.transform(X_low)
X_train_stan_h = scaler.transform(X_high)
X_train_mf, Y_train_mf = convert_xy_lists_to_arrays([X_train_stan_l, X_train_stan_h], [y_low[:,None], y_high[:,None] ])

# Construct a non-linear multi-fidelity model
base_kernel = GPy.kern.RBF

kernels = make_non_linear_kernels(base_kernel, 2, X_train_mf.shape[1] - 1, ARD=True)
nonlin_mf_model = NonLinearMultiFidelityModel(X_train_mf, Y_train_mf, n_fidelities=2, kernels=kernels, 
                                            verbose=True, optimization_restarts=5)
for m in nonlin_mf_model.models:
    m.Gaussian_noise.variance.fix(1e-6)

# Fit the model
nonlin_mf_model.optimize()
for m in nonlin_mf_model.models:
    m.Gaussian_noise.variance.unfix()
nonlin_mf_model.optimize()

#################################
# 1. Plot posterior mean function
# of g_high for MF-GP-nlow500
#################################

#create plot
cm = 1/2.54
fig, ax = plt.subplots(nrows=2, ncols=5, figsize=[18.0*cm,10*cm], dpi=600)

#create test points
n_x = 50
n_y = 50
x = np.linspace(500, 1000, n_x)
y = np.linspace(500, 1000, n_y)
xx, yy = np.meshgrid(x,y)

#loop over different theta values
for i in range(5):
    for j in range(2):
        theta = (j*5+i)*5

        #create test data and standardise
        X_test = np.array([xx.flatten(),yy.flatten(),theta*np.ones(n_x*n_y)]).transpose()
        X_test_stan = scaler.transform(X_test)
        X_test_l = np.concatenate([np.atleast_2d(X_test_stan), np.zeros((X_test_stan.shape[0], 1))], axis=1)
        X_test_h = np.concatenate([np.atleast_2d(X_test_stan), np.ones((X_test_stan.shape[0], 1))], axis=1)

        #make predictions
        lf_mean_mf_model, lf_var_mf_model = nonlin_mf_model.predict(X_test_l)
        hf_mean_mf_model, hf_var_mf_model = nonlin_mf_model.predict(X_test_h)
        hf_mean_mf_model = np.reshape(hf_mean_mf_model.flatten(),(n_x,n_y))
        lf_mean_mf_model = np.reshape(lf_mean_mf_model.flatten(),(n_x,n_y))
        lf_var_mf_model = np.reshape(lf_var_mf_model.flatten(),(n_x,n_y))
        hf_var_mf_model = np.reshape(hf_var_mf_model.flatten(),(n_x,n_y))
        lf_std_mf_model = np.sqrt(lf_var_mf_model)
        hf_std_mf_model = np.sqrt(hf_var_mf_model)

        if i+j==0:
            pcm = ax[j, i].pcolormesh(xx/100, yy/100, hf_mean_mf_model, vmin=0.5, vmax=0.8, rasterized=True)
        else:
           ax[j, i].pcolormesh(xx/100, yy/100, hf_mean_mf_model, vmin=0.5, vmax=0.8, rasterized=True)
        if i==0:
            ax[j, i].set_ylabel(r'$S_y/D$')
        if j==1:
            ax[j, i].set_xlabel(r'$S_x/D$')
        ax[j, i].set_title(f'{string.ascii_lowercase[j*5+i]}) '+r'$\theta=$'+f'{(j*5+i)*5}'+r'$^o$', loc='left')
        ax[j, i].set_aspect('equal')

plt.tight_layout()
plt.subplots_adjust(hspace=0.3)
cbar = fig.colorbar(pcm, ax=ax.ravel().tolist(), location='bottom')
cbar.solids.set_rasterized(True)
cbar.set_label(r'$\overline{m}_{\sigma^2}$')
plt.savefig('figures/MF-GP-nlow500_posterior_mean_hf.png')

#################################
# 2. Plot posterior covariance function
# of g_high for MF-GP-nlow500
#################################

#create plot
cm = 1/2.54
fig, ax = plt.subplots(nrows=2, ncols=5, figsize=[18.0*cm,10*cm], dpi=600)

#loop over different theta values
for i in range(5):
    for j in range(2):
        theta = (j*5+i)*5

        #create test data and standardise
        X_test = np.array([xx.flatten(),yy.flatten(),theta*np.ones(n_x*n_y)]).transpose()
        X_test_stan = scaler.transform(X_test)
        X_test_l = np.concatenate([np.atleast_2d(X_test_stan), np.zeros((X_test_stan.shape[0], 1))], axis=1)
        X_test_h = np.concatenate([np.atleast_2d(X_test_stan), np.ones((X_test_stan.shape[0], 1))], axis=1)

        #make predictions
        lf_mean_mf_model, lf_var_mf_model = nonlin_mf_model.predict(X_test_l)
        hf_mean_mf_model, hf_var_mf_model = nonlin_mf_model.predict(X_test_h)
        hf_mean_mf_model = np.reshape(hf_mean_mf_model.flatten(),(n_x,n_y))
        lf_mean_mf_model = np.reshape(lf_mean_mf_model.flatten(),(n_x,n_y))
        lf_var_mf_model = np.reshape(lf_var_mf_model.flatten(),(n_x,n_y))
        hf_var_mf_model = np.reshape(hf_var_mf_model.flatten(),(n_x,n_y))
        lf_std_mf_model = np.sqrt(lf_var_mf_model)
        hf_std_mf_model = np.sqrt(hf_var_mf_model)

        if i+j==0:
            pcm = ax[j, i].pcolormesh(xx/100, yy/100, hf_std_mf_model, vmin=0, vmax=0.03, rasterized=True)
        else:
           ax[j, i].pcolormesh(xx/100, yy/100, hf_std_mf_model, vmin=0, vmax=0.03, rasterized=True)
        if i==0:
            ax[j, i].set_ylabel(r'$S_y/D$')
        if j==1:
            ax[j, i].set_xlabel(r'$S_x/D$')
        ax[j, i].set_title(f'{string.ascii_lowercase[j*5+i]}) '+r'$\theta=$'+f'{(j*5+i)*5}'+r'$^o$', loc='left')
        ax[j, i].set_aspect('equal')

plt.tight_layout()
plt.subplots_adjust(hspace=0.3)
cbar = fig.colorbar(pcm, ax=ax.ravel().tolist(), location='bottom')
cbar.solids.set_rasterized(True)
cbar.set_label(r'$\sqrt{\overline{k}_{\sigma^2}}$')
plt.savefig('figures/MF-GP-nlow500_posterior_std_hf.png')
