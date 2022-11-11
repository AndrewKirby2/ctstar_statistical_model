import numpy as np
import GPy
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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

ctstar_statistical_model = np.zeros(50)
mae = np.zeros(30)
noise_levels = np.logspace(-7,-1,30)

for j in range(30):

    noise = noise_levels[j]

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
        model = GPy.models.GPRegression(X_train_stan,y_train-ctstar_wake_model[train_index,2][:,None],kernel)
        model.Gaussian_noise = noise_levels[j]
        model.Gaussian_noise.fix()
        model.optimize_restarts(num_restarts = 3)

        #make predictions
        X_test_stan = scaler.transform(X_test)
        y_pred, var = model.predict(X_test_stan)
        #store prediction ctstar values
        ctstar_statistical_model[i] = y_pred+ctstar_wake_model[test_index,2]
    
    mae[j] = 100*np.mean(np.abs(ctstar_statistical_model[:]-training_data[:,3]))/0.75

plt.plot(np.logspace(-7,-1,30),mae)
plt.xscale('log')
plt.ylabel('LOOCV MAE (%)')
plt.xlabel('Gaussian noise variance')
plt.savefig('gaussian_noise_sensitvity.png')
