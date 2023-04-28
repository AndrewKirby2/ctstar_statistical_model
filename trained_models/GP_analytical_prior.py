"""Make predictions of C_T^* using
GP-analytical-prior
"""

import pickle
import numpy as np

def GP_analytical_prior(S_x, S_y, theta):
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

    with open('trained_models/GP-analytical-prior.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('trained_models/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    X = np.array([S_x, S_y, theta]).transpose()

    X_stan = scaler.transform(X.reshape(1,-1))

    y, var = model.predict(X_stan)

    #add prior mean
    ctstar = float(y + 0.75)
    var = float(var)

    return ctstar, var
