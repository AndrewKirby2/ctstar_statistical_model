"""Make predictions of C_T^* using
MF-GP-nlow250
"""

import pickle
import numpy as np

def MF_GP_nlow250(S_x, S_y, theta):
    """  Predicts C_T^* for turbine layout
    using GP-analytical-prior model

    Parameters
    ----------
    S_x : float
        Distance between turbines in x direction normalised by turbine diameter
    S_y : float
        Distance between turbines in y direction normalised by turbine diameter
    theta : float
        Angle between wind direction and x axis in degrees

    Returns
    -------
    ctstar : float
        predictions of `internal' turbine thrust coefficient ctstar
    var : float
        predictive variance
    """

    assert type(S_x)==float and type(S_y)==float and type(theta)==float, "Use single float value for turbine layout parameters"
    assert S_x >= 5 and S_x <= 10, "5 <= S_x <= 10, note spacing is normalised by turbine diameter"
    assert S_y >= 5 and S_y <= 10, "5 <= S_y <= 10, note spacing is normalised by turbine diameter"
    assert theta >= 0 and theta <= 45, "0 <= theta <= 45, note wind direction in degrees"

    #load saved data and scaler
    with open('trained_models/MF-GP-nlow250.pkl', 'rb') as file:
        nonlin_mf_model = pickle.load(file)
    with open('trained_models/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    X = np.array([S_x, S_y, theta]).transpose()

    X_stan = scaler.transform(X.reshape(1,-1))

    #predicting high fidelity C_{T,LES}^*
    X_l_fidelity = np.concatenate([np.atleast_2d(X_stan), np.zeros((X_stan.shape[0], 1))], axis=1)
    X_h_fidelity = np.concatenate([np.atleast_2d(X_stan), np.ones((X_stan.shape[0], 1))], axis=1)
    print(X_h_fidelity)
    y_l, var_l = nonlin_mf_model.predict(X_l_fidelity)
    y_h, var_h = nonlin_mf_model.predict(X_h_fidelity)

    #add prior mean
    ctstar = float(y_h)
    var = float(var_h)

    return ctstar, var
