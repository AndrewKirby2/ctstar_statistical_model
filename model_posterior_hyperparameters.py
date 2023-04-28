""" Calculate posterior hyperparameters
 for each model
"""

import numpy as np
import GPy
from sklearn.preprocessing import StandardScaler
import emukit.multi_fidelity
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models.non_linear_multi_fidelity_model import make_non_linear_kernels, NonLinearMultiFidelityModel
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays

#load LES data
training_data = np.genfromtxt('data/LES_training_data.csv', delimiter=',')
#remove header
training_data = np.delete(training_data, 0, 0)
training_data = np.delete(training_data, 0, 1)
X = training_data[:,:3]
y = training_data[:,3]

#load the wake model prior mean
ctstar_wake_model = np.genfromtxt('data/ctstar_wake_model.csv', delimiter=',')
#remove header
ctstar_wake_model = np.delete(ctstar_wake_model, 0, 0)
ctstar_wake_model = np.delete(ctstar_wake_model, 0, 1)

#creating training data
X_train = X
y_train = y[:,None]

#standardise the feature set of the training and test data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_stan = scaler.transform(X_train)

#array to store standard GP posterior hyperparameters
gp_post_hyp_param = np.zeros((5,4))

###########################
# 1. GP-analytical-prior
###########################

#create GPy kernel
kernel = GPy.kern.RBF(input_dim=3,ARD=True)

#train GP model
model = GPy.models.GPRegression(X_train_stan,y_train-0.75,kernel)
model.optimize_restarts(num_restarts = 10, messages=False)
 
gp_post_hyp_param[0,0] = kernel.variance.values
gp_post_hyp_param[0,1:] = kernel.lengthscale.values

###########################
# 2. GP-wakeTI{x}-prior
###########################

ti = [1, 5, 10, 15]

#loop over different ambient TI for
#wake model prior mean
for j in range(4):

    #create GPy kernel
    kernel = GPy.kern.RBF(input_dim=3,ARD=True)

    #train GP model
    model = GPy.models.GPRegression(X_train_stan,y_train-ctstar_wake_model[:,j][:,None],kernel)
    model.optimize_restarts(num_restarts = 10)

    gp_post_hyp_param[j+1,0] = kernel.variance.values
    gp_post_hyp_param[j+1,1:] = kernel.lengthscale.values

###########################
# 3. MF-GP-nlow{x}
###########################

#array to store mf GP posterior hyperparameters
mf_gp_post_hyp_param = np.zeros((3,14))

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

    #scale the data
    X_train_stan_l = scaler.transform(X_low)
    X_train_stan_h = scaler.transform(X_train)
    X_train_mf, Y_train_mf = convert_xy_lists_to_arrays([X_train_stan_l, X_train_stan_h], [y_low[:,None], y[:,None] ])

    # Construct a non-linear multi-fidelity model
    base_kernel = GPy.kern.RBF

    #define model
    kernels = make_non_linear_kernels(base_kernel, 2, X_train_mf.shape[1] - 1, ARD=True)
    nonlin_mf_model = NonLinearMultiFidelityModel(X_train_mf, Y_train_mf, n_fidelities=2, kernels=kernels, 
                                            verbose=True, optimization_restarts=5)
    for m in nonlin_mf_model.models:
        m.Gaussian_noise.variance.fix(1e-6)

    nonlin_mf_model.optimize()
    for m in nonlin_mf_model.models:
        m.Gaussian_noise.variance.unfix()
    nonlin_mf_model.optimize()

    mf_gp_post_hyp_param[j,0] = kernels[0].variance
    mf_gp_post_hyp_param[j,1:4] = kernels[0].lengthscale
    mf_gp_post_hyp_param[j,4] = kernels[1].mul.scale_kernel_fidelity2.variance
    mf_gp_post_hyp_param[j,5:8] = kernels[1].mul.scale_kernel_fidelity2.lengthscale
    mf_gp_post_hyp_param[j,8] = kernels[1].mul.previous_fidelity_fidelity2.variance
    mf_gp_post_hyp_param[j,9] = kernels[1].mul.previous_fidelity_fidelity2.lengthscale
    mf_gp_post_hyp_param[j,10] = kernels[1].bias_kernel_fidelity2.variance
    mf_gp_post_hyp_param[j,11:14] = kernels[1].bias_kernel_fidelity2.lengthscale

print(gp_post_hyp_param)
print(mf_gp_post_hyp_param)