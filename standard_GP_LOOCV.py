""" Perform LOOCV of standard GP approach
using different prior means
"""

import numpy as np
import GPy
from sklearn.preprocessing import StandardScaler

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
#array to store ctstar predictions
ctstar_statistical_model = np.zeros((50,5))

##############################################
# 1. Using wake model prior mean
# note that index [:,0] means ambient TI = 1%
# [:,1] ambient TI = 5%
# [:,2] ambient TI = 10%
# [:,3] ambient TI = 15%
###############################################
ti = [1, 5, 10, 15]

#loop over different ambient TI for
#wake model prior mean
for j in range(4):

    #array to store posterior hyperparameters
    post_hyp_param = np.zeros((50,4))

    #loop over data points to perform LOOCV
    for i in range(50):

        #create training and testing data
        train_index = list(filter(lambda x: x!= i, range(50)))
        test_index = i

        X_train = X[train_index,:]
        X_test = X[test_index,:][None,:]
        y_train = y[train_index][:,None]
        y_test = y[test_index]

        #standardise the feature set of the training and test data
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_stan = scaler.transform(X_train)

        #create GPy kernel
        kernel = GPy.kern.RBF(input_dim=3,ARD=True)

        #train GP model
        model = GPy.models.GPRegression(X_train_stan,y_train-ctstar_wake_model[train_index,j][:,None],kernel)
        model.optimize_restarts(num_restarts = 10)

        post_hyp_param[i,0] = kernel.variance
        post_hyp_param[i,1:4] = kernel.lengthscale

        #make predictions
        X_test_stan = scaler.transform(X_test)
        y_pred, var = model.predict(X_test_stan)
        #store prediction ctstar values
        ctstar_statistical_model[i,j] = y_pred+ctstar_wake_model[test_index,j]

        #save posterior hyperparameters to csv file
        np.savetxt(f'posterior_hyperparameters/GP-wake-TI{ti[j]}-prior.csv', post_hyp_param, delimiter=',',
        header = 'Variance, Lengthscale 1 (S_x), Lengthscale 2 (S_y), Lengthscale 3 (theta)')

###########################
# 2. Using analytical model
# prior mean
###########################

#array to store posterior hyperparameters
post_hyp_param = np.zeros((50,4))

for i in range(50):

    #create training and testing data
    train_index = list(filter(lambda x: x!= i, range(50)))
    test_index = i

    X_train = X[train_index,:]
    X_test = X[test_index,:][None,:]
    y_train = y[train_index][:,None]
    y_test = y[test_index]

    #standardise the feature set of the training and test data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_stan = scaler.transform(X_train)

    #create GPy kernel
    kernel = GPy.kern.RBF(input_dim=3,ARD=True)

    #train GP model
    model = GPy.models.GPRegression(X_train_stan,y_train-0.75,kernel)
    model.optimize_restarts(num_restarts = 10)

    post_hyp_param[i,0] = kernel.variance
    post_hyp_param[i,1:4] = kernel.lengthscale

    #make predictions
    X_test_stan = scaler.transform(X_test)
    y_pred, var = model.predict(X_test_stan)
    ctstar_statistical_model[i,4] = y_pred+0.75

    #save posterior hyperparameters to csv file
    np.savetxt(f'posterior_hyperparameters/GP-wake-analytical-prior.csv', post_hyp_param, delimiter=',',
    header = 'Variance, Lengthscale 1 (S_x), Lengthscale 2 (S_y), Lengthscale 3 (theta)')

# print results of standard GP models

for j in range(4):
    print('Wake model prior mean ambient TI='+str(ti[j])+'%')
    print('MAE = '+str(100*np.mean(np.abs(ctstar_statistical_model[:,j]-training_data[:,3]))/0.75)
    +'%       Max error = '+str(100*np.max(np.abs(ctstar_statistical_model[:,j]-training_data[:,3]))/0.75))
    print('MAE = '+str(np.mean(np.abs(ctstar_statistical_model[:,j]-training_data[:,3]))))

print('Analytical model prior mean')
print('MAE ='+str(100*np.mean(np.abs(ctstar_statistical_model[:,4]-training_data[:,3]))/0.75)
    +'%       Max error = '+str(100*np.max(np.abs(ctstar_statistical_model[:,4]-training_data[:,3]))/0.75))
print('MAE ='+str(np.mean(np.abs(ctstar_statistical_model[:,4]-training_data[:,3]))))
