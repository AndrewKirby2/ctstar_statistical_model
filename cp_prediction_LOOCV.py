"""1. Correct LES wind speed 
2. Calculate Cp from LES data different
wind farm "extractability"
3. Predict Cp using two scale momentum theory
4. Plot results
"""

import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

#wind farm parameters
#momentum `extractability' factor
zeta=[0,5,10,15,20,25]
#bottom friction exponent
gamma=2

#arrays to store result
cp_finite = np.zeros((50,6))
effective_area_ratio = np.zeros(50)
cp_statistical_model = np.zeros((50,6))

#load statistical model predictions of C_T^*
ctstar_statistical_model = np.genfromtxt('ctstar_nonlin_statistical_model_500.csv', delimiter=',')

#load LES data
training_data = np.genfromtxt('LES_training_data.csv', delimiter=',')
#remove header
training_data = np.delete(training_data, 0, 0)
training_data = np.delete(training_data, 0, 1)

#note correction factor N^2 already applied!
ct_star = training_data[:,3]
beta = training_data[:,5]
#note correction factor N^3 already applied!
cp = training_data[:,7]
cp_corrected = np.zeros((50))
beta_corrected = np.zeros((50))

################################
#1. Correct LES wind speed
################################

for run_no in range(50):

    #calculate effective area ratio
    C_f0 = 0.28641758**2/(0.5*10.10348311**2)
    A = np.pi/4
    S = training_data[run_no,0]*training_data[run_no,1]
    area_ratio = A/S
    effective_area_ratio[run_no] = area_ratio/C_f0

    #calculate beta_fine_theory
    def NDFM(beta):
        """ Non-dimensional farm momentum
        equation (see Nishino 2020)
        """
    #use ct_star to predict beta_fine_theory
        return ct_star[run_no]*effective_area_ratio[run_no]*beta**2 + beta**gamma - 1

    beta_fine_theory = sp.bisect(NDFM,0,1)

    #calculate beta_coarse_theory
    def NDFM(beta):
        """ Non-dimensional farm momentum
        equation (see Nishino 2020)
        """
    #use ct_star to predict beta_fine_theory
        return (ct_star[run_no]/ 0.8037111)*effective_area_ratio[run_no]*beta**2 + beta**gamma - 1

    beta_coarse_theory = sp.bisect(NDFM,0,1)

    #correct Cp values recorded by LES
    cp_corrected[run_no] = cp[run_no]*(beta_fine_theory/beta_coarse_theory)**3
    #correct beta value recorded by LES
    beta_corrected[run_no] = beta[run_no]*(beta_fine_theory/beta_coarse_theory)


#repeat for different zeta values
for i in range(6):

    #############################################
    # 2. Calculate Cp from LES data for a finite
    # wind farm
    #############################################

    #calculate adjusted Cp and effective area ratio
    #for each wind farm LES
    for run_no in range(50):
        U_F = beta_corrected[run_no]*10.10348311
        U_F0 = 10.10348311

        #coefficients of quadratic formula to solve
        a = 1/U_F**2
        b = zeta[i]/U_F0
        c = -zeta[i] - 1

        U_Fprime = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
    
        cp_finite[run_no,i] = cp_corrected[run_no]*(U_Fprime/U_F)**3

    #############################################
    # 3. Predict Cp using two-scale momentum
    # theory
    #############################################

    #predict Cp for each wind farm LES
    for run_no in range(50):

        def NDFM(beta):
            """ Non-dimensional farm momentum
            equation (see Nishino 2020)
            """
	    #use ct_star to predict beta
	    #ctstar predictions from LOOCV
            return ctstar_statistical_model[run_no]*effective_area_ratio[run_no]*beta**2 + beta**gamma - 1 -zeta[i]*(1-beta)

        beta_theory = sp.bisect(NDFM,0,1)
        cp_statistical_model[run_no,i] = ctstar_statistical_model[run_no]**1.5 * beta_theory**3 * 1.33**-0.5

    print(mean_absolute_error(cp_finite[:,i], cp_statistical_model[:,i])/0.563205)

#############################################
# 4. Plot results
#############################################
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=[5.33,6.6], dpi=600)
ax[0,0].scatter(effective_area_ratio, cp_statistical_model[:,0], s=30, marker='x', c='r')
ax[0,0].scatter(effective_area_ratio, cp_finite[:,0], s=30, marker='^', facecolors='none', edgecolors='k')
ax[0,0].set_xlabel(r'$\lambda/C_{f0}$')
ax[0,0].set_ylabel(r'$C_p$')
ax[0,0].set_title('a)', loc='left')

ax[0,1].scatter(effective_area_ratio, cp_statistical_model[:,1], s=30, marker='x', c='r')
ax[0,1].scatter(effective_area_ratio, cp_finite[:,1], s=30, marker='^', facecolors='none', edgecolors='k')
ax[0,1].set_xlabel(r'$\lambda/C_{f0}$')
ax[0,1].set_ylabel(r'$C_{p,finite}$')
ax[0,1].set_title('b)', loc='left')

ax[1,0].scatter(effective_area_ratio, cp_statistical_model[:,2], s=30, marker='x', c='r')
ax[1,0].scatter(effective_area_ratio, cp_finite[:,2], s=30, marker='^', facecolors='none', edgecolors='k')
ax[1,0].set_xlabel(r'$\lambda/C_{f0}$')
ax[1,0].set_ylabel(r'$C_{p.finite}$')
ax[1,0].set_title('c)', loc='left')

ax[1,1].scatter(effective_area_ratio, cp_statistical_model[:,3], s=30, marker='x', c='r')
ax[1,1].scatter(effective_area_ratio, cp_finite[:,3], s=30, marker='^', facecolors='none', edgecolors='k')
ax[1,1].set_xlabel(r'$\lambda/C_{f0}$')
ax[1,1].set_ylabel(r'$C_{p,finite}$')
ax[1,1].set_title('d)', loc='left')

ax[2,0].scatter(effective_area_ratio, cp_statistical_model[:,4], s=30, marker='x', c='r')
ax[2,0].scatter(effective_area_ratio, cp_finite[:,4], s=30, marker='^', facecolors='none', edgecolors='k')
ax[2,0].set_xlabel(r'$\lambda/C_{f0}$')
ax[2,0].set_ylabel(r'$C_{p,finite}$')
ax[2,0].set_title('e)', loc='left')

ax[2,1].scatter(effective_area_ratio, cp_statistical_model[:,5], s=30, marker='x', c='r')
ax[2,1].scatter(effective_area_ratio, cp_finite[:,5], s=30, marker='^', facecolors='none', edgecolors='k')
ax[2,1].set_xlabel(r'$\lambda/C_{f0}$')
ax[2,1].set_ylabel(r'$C_{p,finite}$')
ax[2,1].set_title('f)', loc='left')

plt.tight_layout()

plt.savefig('LES_cp_results.png', bbox_inches='tight')