"""Calculate C_T^* as a function
of S_x, S_y, theta
"""

import numpy as np
import diversipy.hycusampling as dp 
from py_wake import YZGrid
from py_wake.turbulence_models import CrespoHernandez
from py_wake.superposition_models import LinearSum
from py_wake.rotor_avg_models import GQGridRotorAvg
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.wind_turbines import OneTypeWindTurbines
from py_wake.deficit_models import NiayifarGaussianDeficit
from py_wake.site import UniformSite

def wake_model(S_x, S_y, theta, ti):
    """Use a low order wake model to
    calculate C_T^*

    :param S_x: float distance between turbines in the x direction (metres)
    :param S_y: float distance between turbines in the y direction (metres)
    :param theta: float wind direction with respect to x direction (degrees)
    :param ti:  float ambient turbulence intensity (%)

    :returns ct_star: float local turbine thrust coefficient (dimensionless)
    """

    #estimate thrust coefficient ct
    ct_prime = 1.33
    a = ct_prime/(4 + ct_prime)
    ct = 4*a*(1-a)

    #define a farm site with an ambient turbulence intensity
    site = UniformSite([1],ti/100)

    #calculate turbine coordinates
    x = np.hstack((np.arange(0, -10000, -S_x),np.arange(S_x, 1500, S_x)))
    y = np.hstack((np.arange(0, 8500, S_y),np.arange(-S_y,-2000,-S_y)))
    xx, yy = np.meshgrid(x,y)
    x = xx.flatten()
    y = yy.flatten()

    #only consider turbines 10km upstream or 1km in the cross stream direction
    streamwise_cond = -x*np.cos(theta*np.pi/180) +y*np.sin(theta*np.pi/180) < 10000
    spanwise_cond = abs(-y*np.cos(theta*np.pi/180) - x*np.sin(theta*np.pi/180)) < 2000
    total_cond = np.logical_and(streamwise_cond, spanwise_cond)
    x_rot = x[total_cond]
    y_rot = y[total_cond]

    #create ideal turbines with constant thrust coefficients
    my_wt = OneTypeWindTurbines(name='MyWT',
                           diameter=100,
                           hub_height=100,
                           ct_func=lambda ws : np.interp(ws, [0, 30], [ct, ct]),
                           power_func=lambda ws : np.interp(ws, [0, 30], [2, 2]),
                           power_unit='kW')
    windTurbines = my_wt

    #select models to calculate wake deficits behind turbines
    wake_deficit = PropagateDownwind(site, windTurbines, NiayifarGaussianDeficit(a=[0.38, 4e-3],use_effective_ws=True),
                                superpositionModel=LinearSum(), rotorAvgModel = GQGridRotorAvg(4,3),
                                turbulenceModel=CrespoHernandez())
    
    #run wind farm simulation
    simulationResult = wake_deficit(x_rot, y_rot, ws=10, wd=270+theta)

    #calculate turbine disk velocity
    U_T = (1-a)*simulationResult.WS_eff[0]

    #calculate velocity in wind farm layer (0-250m above the surface)
    U_F = 0
    for i in np.linspace(-S_x, 0, 200):
        grid = YZGrid(x = i, y = np.linspace(-S_y/2,S_y/2,200),
                        z = np.linspace(0,250,20))
        flow_map = simulationResult.flow_map(grid=grid, ws=10, wd=270+theta)
        U_F += np.mean(flow_map.WS_eff)
    U_F = U_F/200

    #calculate local turbine thrust coefficient
    ct_star = float(ct_prime*(U_T/U_F)**2)
    return ct_star

##load LES training data
#training_data = np.genfromtxt('training_data.csv', delimiter=',')
#empty array to store wake model results
#ctstar_wake_model = np.zeros((50,6))
#ctstar_wake_model[:,:4] = np.genfromtxt('ctstar_wake_model.csv', delimiter=',')
#print(ctstar_wake_model)
#array of ambient turbulence intensity to loop over
#ti = [1,5,10,15,20,25]

#loop over TI levels
#for i in range(4,6):
#    print("Turbulence intensity ",ti[i],"%")
    #loop over each wind case
#    for j in range(50):
#        print("Wind farm case ",j)
#        ctstar_wake_model[j,i] = wake_model(training_data[j,0], training_data[j,1], training_data[j,2], ti[i])
#        print(ctstar_wake_model[j,i])

#np.savetxt('ctstar_wake_model.csv', ctstar_wake_model, delimiter=',')

#experimental design for low fidelity observations
design_lofi = dp.maximin_reconstruction(250,3)
design_lofi[:,:2] = 500 + 500*design_lofi[:,:2]
design_lofi[:,2] = 45*design_lofi[:,2]
print(np.shape(design_lofi))

wake_model_results = np.zeros((250,4))
wake_model_results[:,:3] = design_lofi
for i in range(250):
    print(i)
    wake_model_results[i,3] = wake_model(wake_model_results[i,0], wake_model_results[i,1], wake_model_results[i,2], 10)

np.savetxt('wake_model_results_250.csv', wake_model_results, delimiter=',')