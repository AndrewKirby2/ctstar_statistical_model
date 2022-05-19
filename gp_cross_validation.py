import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut

#load the LES training data
training_data = np.genfromtxt('training_data.csv', delimiter = ',')
X = training_data[:,:3]
y = training_data[:,3]

#load the wake model prior mean
ctstar_wake_model = np.genfromtxt('ctstar_wake_model.csv', delimiter=',')
prior_mean = ctstar_wake_model[:,2]

#array to hold results
error = np.zeros(50)

loo = LeaveOneOut()
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(test_index)

    #standardise the feature set of the training and test data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_stan = scaler.transform(X_train)


    #create kernel for Gaussian Process Regression
    kernel = 1.0 ** 2 * RBF(length_scale=[1.,1.,1.]) + WhiteKernel(noise_level=1e-3, noise_level_bounds=[1e-10,1])
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50)


    #fit GP and make predictions
    gp.fit(X_train_stan,y_train-prior_mean[train_index])
    print(gp.kernel_)

    X_test_stan = scaler.transform(X_test)

    print(mean_absolute_error(y_test, gp.predict(X_test_stan) + 
      prior_mean[test_index])/0.75)
    error[test_index] = mean_absolute_error(y_test, gp.predict(X_test_stan) + 
      prior_mean[test_index])/0.75
    
print(np.mean(error))
print(np.max(error))
