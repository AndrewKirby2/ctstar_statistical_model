"""Plot posterior variance
of GP-wake-TI10-prior mean
i.e., figure 6
"""

import numpy as np
import GPy
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import string

#create plot
cm = 1/2.54
fig, ax = plt.subplots(nrows=2, ncols=5, figsize=[18.0*cm,10*cm], dpi=600)

#load the LES training data
training_data = np.genfromtxt('training_data.csv', delimiter = ',')
X = training_data[:,:3]
y = training_data[:,3]

#load the wake model prior mean
ctstar_wake_model = np.load('data/standard_GP_ctstar_predictions.npy')
#load wake model data for TI=10%
ctstar_wake_model_TI10 = ctstar_wake_model[:,2]

#scale input data
scaler = StandardScaler()
scaler.fit(X)
X_train_stan = scaler.transform(X)

#create GPy kernel
kernel = GPy.kern.RBF(input_dim=3,ARD=True) + GPy.kern.White(input_dim=3)

#train GP model
model = GPy.models.GPRegression(X_train_stan,y-ctstar_wake_model[:,2][:,None],kernel)
model.optimize_restarts(num_restarts = 10)

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

        #make predictions
        y_pred, var = model.predict(X_test_stan)
        std = np.sqrt(var)
        std = np.reshape(std.flatten(),(n_x,n_y))

        if i+j==0:
            pcm = ax[j, i].pcolormesh(xx/100, yy/100, std, vmin=0, vmax=0.03, rasterized=True)
        else:
           ax[j, i].pcolormesh(xx/100, yy/100, std, vmin=0, vmax=0.03, rasterized=True)
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
plt.savefig('figures/GP-wake-TI1-prior_posterior_variance.png')