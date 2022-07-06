import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, max_error
from sklearn.model_selection import LeaveOneOut

#load the LES training data
training_data = np.genfromtxt('training_data.csv', delimiter = ',')
X = training_data[:,:3]
y = training_data[:,3]

#load the wake model prior mean
ctstar_wake_model = np.genfromtxt('ctstar_wake_model.csv', delimiter=',')

#array to hold statistical model results
ctstar_statistical_model = np.zeros((50,6))

for i in range(6):
  print(i)
  loo = LeaveOneOut()
  for train_index, test_index in loo.split(X):
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]

      #standardise the feature set of the training and test data
      scaler = StandardScaler()
      scaler.fit(X_train)
      X_train_stan = scaler.transform(X_train)


      #create kernel for Gaussian Process Regression
      kernel = 1.0 ** 2 * RBF(length_scale=[1.,1.,1.]) + WhiteKernel(noise_level=1e-3, noise_level_bounds=[1e-10,1])
      gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50)


      #fit GP and make predictions
      gp.fit(X_train_stan,y_train-ctstar_wake_model[train_index,i])

      X_test_stan = scaler.transform(X_test)

      ctstar_statistical_model[test_index,i] = gp.predict(X_test_stan) + ctstar_wake_model[test_index,i]
for i in range(6):
  print(mean_absolute_error(ctstar_statistical_model[:,i], training_data[:,3])/0.75)
  print(max_error(ctstar_statistical_model[:,i], training_data[:,3])/0.75)

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=[5.33,8])
ax[0,0].scatter(training_data[:,3], ctstar_wake_model[:,0], label = 'Wake model')
ax[0,0].scatter(training_data[:,3], ctstar_statistical_model[:,0], label = 'Statistical model')
ax[0,0].plot([0.55,0.8],[0.55,0.8], c='k')
ax[0,0].set_xlim([0.55,0.8])
ax[0,0].set_ylim([0.45,0.8])
ax[0,0].set_ylabel(r'$C_T^*$')
ax[0,0].set_xlabel(r'$C_{T,LES}^*$')

ax[0,1].scatter(training_data[:,3], ctstar_wake_model[:,1], label = 'Wake model')
ax[0,1].scatter(training_data[:,3], ctstar_statistical_model[:,1], label = 'Statistical model')
ax[0,1].plot([0.55,0.8],[0.55,0.8], c='k')
ax[0,1].set_xlim([0.55,0.8])
ax[0,1].set_ylim([0.45,0.8])
ax[0,1].set_ylabel(r'$C_T^*$')
ax[0,1].set_xlabel(r'$C_{T,LES}^*$')

ax[1,0].scatter(training_data[:,3], ctstar_wake_model[:,2], label = 'Wake model')
ax[1,0].scatter(training_data[:,3], ctstar_statistical_model[:,2], label = 'Statistical model')
ax[1,0].plot([0.55,0.8],[0.55,0.8], c='k')
ax[1,0].set_xlim([0.55,0.8])
ax[1,0].set_ylim([0.45,0.8])
ax[1,0].set_ylabel(r'$C_T^*$')
ax[1,0].set_xlabel(r'$C_{T,LES}^*$')

ax[1,1].scatter(training_data[:,3], ctstar_wake_model[:,3], label = 'Wake model')
ax[1,1].scatter(training_data[:,3], ctstar_statistical_model[:,3], label = 'Statistical model')
ax[1,1].plot([0.55,0.8],[0.55,0.8], c='k')
ax[1,1].set_xlim([0.55,0.8])
ax[1,1].set_ylim([0.45,0.8])
ax[1,1].set_ylabel(r'$C_T^*$')
ax[1,1].set_xlabel(r'$C_{T,LES}^*$')

ax[2,0].scatter(training_data[:,3], ctstar_wake_model[:,4], label = 'Wake model')
ax[2,0].scatter(training_data[:,3], ctstar_statistical_model[:,4], label = 'Statistical model')
ax[2,0].plot([0.55,0.8],[0.55,0.8], c='k')
ax[2,0].set_xlim([0.55,0.8])
ax[2,0].set_ylim([0.45,0.8])
ax[2,0].set_ylabel(r'$C_T^*$')
ax[2,0].set_xlabel(r'$C_{T,LES}^*$')

ax[2,1].scatter(training_data[:,3], ctstar_wake_model[:,5], label = 'Wake model')
ax[2,1].scatter(training_data[:,3], ctstar_statistical_model[:,5], label = 'Statistical model')
ax[2,1].plot([0.55,0.8],[0.55,0.8], c='k')
ax[2,1].set_xlim([0.55,0.8])
ax[2,1].set_ylim([0.45,0.8])
ax[2,1].set_ylabel(r'$C_T^*$')
ax[2,1].set_xlabel(r'$C_{T,LES}^*$')

fig.legend(bbox_to_anchor=(0.5,-0.1), loc="lower center", 
                bbox_transform=fig.transFigure, ncol=2)
plt.tight_layout()
plt.savefig('statistical_model_results.png')