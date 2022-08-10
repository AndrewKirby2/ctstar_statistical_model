import numpy as np
from matplotlib import pyplot as plt
import GPy
import deepgp
from sklearn.preprocessing import StandardScaler

#load the LES training data
training_data = np.genfromtxt('training_data.csv', delimiter = ',')
X = training_data[:,:3]
y = training_data[:,3]

num_hidden = 1
latent_dim = 3

for i in range(50):

    #create training and testing data
    train_index = list(filter(lambda x: x!= i, range(50)))
    test_index = i

    X_train = X[train_index,:]
    X_test = X[test_index,:][None,:]
    y_train = y[train_index][:,None]
    offset = y_train.mean()
    scale = np.sqrt(y_train.var())
    y_hat = (y_train - offset)/scale
    y_test = y[test_index]

    #standardise the feature set of the training and test data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_stan = scaler.transform(X_train)

    kernels = [*[GPy.kern.RBF(latent_dim, ARD=True)]*num_hidden] # hidden kernels
    kernels.append(GPy.kern.RBF(X.shape[1]))

    m_deep = deepgp.DeepGP(
        # this describes the shapes of the inputs and outputs of our latent GPs
        [y_train.shape[1], *[latent_dim]*num_hidden, X_train.shape[1]],
        X = X_train, # training input
        Y = y_hat, # training outout
        inits = [*['PCA']*num_hidden, 'PCA'], # initialise layers
        kernels = kernels,
        num_inducing = 50,
        back_constraint = False
    )

    m_deep.initialize_parameter()
    for layer in m_deep.layers:
        layer.likelihood.variance.constrain_positive(warning=False)
        layer.likelihood.variance = 1. # small variance may cause collapse
    m_deep.optimize(messages=True, max_iters=10000)

    X_test_stan = scaler.transform(X_test)
    y_pred, var = m_deep.predict(X_test_stan)
    print(y_pred+0.75, y[test_index])