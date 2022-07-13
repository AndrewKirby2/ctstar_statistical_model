import numpy as np
import GPy
from IPython.display import display
from sklearn.preprocessing import StandardScaler

#load the LES training data
training_data = np.genfromtxt('training_data.csv', delimiter = ',')
X = training_data[:,:3]
y = training_data[:,3]

#load the wake model prior mean
ctstar_wake_model = np.genfromtxt('ctstar_wake_model.csv', delimiter=',')
ctstar_statistical_model = np.zeros(50)

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
    kernel = GPy.kern.RBF(input_dim=3,ARD=True) + GPy.kern.White(input_dim=3)

    #train GP model
    model = GPy.models.GPRegression(X_train_stan,y_train-ctstar_wake_model[train_index,2][:,None],kernel)
    model.optimize_restarts(num_restarts = 10)

    #make predictions
    X_test_stan = scaler.transform(X_test)
    y_pred, var = model.predict(X_test_stan)
    ctstar_statistical_model[i] = y_pred+ctstar_wake_model[test_index,2]
    print(var)
    print(kernel)
    print(kernel.rbf.lengthscale)

print(np.mean(np.abs(ctstar_statistical_model-training_data[:,3]))/0.75)