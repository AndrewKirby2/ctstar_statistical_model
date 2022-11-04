"""1. Correct LES wind speed 
2. Calculate Cp from LES data different
wind farm "extractability"
3. Predict Cp using two scale momentum theory
4. Plot results
"""

import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

#wind farm parameters
#momentum `extractability' factor
zeta=[0,5,10,15,20,25]
#bottom friction exponent
gamma=2

#arrays to store result
cp_finite = np.zeros((50,6))
effective_area_ratio = np.zeros(50)
cp_statistical_model = np.zeros((50,6))
cp_theory_predictions = np.zeros((50,6))
cp_theory_trend = np.zeros((50,6))
effective_area_ratio_trend = np.linspace(1,20,50)

#load statistical model predictions of C_T^*
ctstar_statistical_model = np.load('data/mf_GP_ctstar_predictions.csv')
#remove header
ctstar_statistical_model = np.delete(ctstar_statistical_model, 0, 0)
ctstar_statistical_model = np.delete(ctstar_statistical_model, 0, 1)
#select predictions from MF-GP-nlow500
ctstar_statistical_model = ctstar_statistical_model[:,1]

#load LES data
training_data = np.genfromtxt('data/LES_training_data.csv', delimiter=',')
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

        #predict cp using statistical model of C_T^* (LOOCV)
        def NDFM(beta):
            """ Non-dimensional farm momentum
            equation (see Nishino 2020)
            """
            return ctstar_statistical_model[run_no]*effective_area_ratio[run_no]*beta**2 + beta**gamma - 1 -zeta[i]*(1-beta)

        beta_theory = sp.bisect(NDFM,0,1)
        cp_statistical_model[run_no,i] = ctstar_statistical_model[run_no]**1.5 * beta_theory**3 * 1.33**-0.5

        #predict C_T^* using analytical model of C_T^*
        def NDFM(beta):
            """ Non-dimensional farm momentum
            equation (see Nishino 2020)
            """
            return 0.75*effective_area_ratio[run_no]*beta**2 + beta**gamma - 1 -zeta[i]*(1-beta)

        beta_theory = sp.bisect(NDFM,0,1)
        cp_theory_predictions[run_no,i] = 0.75**1.5 * beta_theory**3 * 1.33**-0.5

        #save data to plot theoretical trend of C_p
        def NDFM(beta):
            """ Non-dimensional farm momentum
            equation (see Nishino 2020)
            """
            return 0.75*effective_area_ratio_trend[run_no]*beta**2 + beta**gamma - 1 -zeta[i]*(1-beta)

        beta_theory = sp.bisect(NDFM,0,1)
        cp_theory_trend[run_no,i] = 0.75**1.5 * beta_theory**3 * 1.33**-0.5

#print table of MAE
print("Mean Absolute Error")
print('zeta     Analytical model        Statistical model ')
for i in range(6):
    print(zeta[i],"     ", mean_absolute_error(cp_finite[:,i], cp_theory_predictions[:,i])
           ,"      ",  mean_absolute_error(cp_finite[:,i], cp_statistical_model[:,i]))

#print table of MAPE
print("Mean Absolute Percentage Error")
print('zeta     Analytical model (%)    Statistical model (%)')
for i in range(6):
    print(zeta[i],"     ", 100*mean_absolute_percentage_error(cp_finite[:,i], cp_theory_predictions[:,i])
           ,"      ",  100*mean_absolute_percentage_error(cp_finite[:,i], cp_statistical_model[:,i]))

#print table of MAE
print()
print("Mean Absolute Error (normalised by Betz prediction)")
print('zeta     Analytical model (%)    Statistical model (%)')
for i in range(6):
    print(zeta[i],"        ", 100*mean_absolute_error(cp_finite[:,i], cp_theory_predictions[:,i])/0.563205
           ,"      ",  100*mean_absolute_error(cp_finite[:,i], cp_statistical_model[:,i])/0.563205)


np.save('cp_finite.npy', cp_finite)
np.save('cp_statistical_model.npy', cp_statistical_model)
np.save('cp_theory_predictions.npy', cp_theory_predictions)
np.save('cp_theory_trend.npy', cp_theory_trend)
np.save('effective_area_ratio_trend', effective_area_ratio_trend)

#############################################
# 4. Plot results
#############################################
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=[5.33,6.6], dpi=600)
ax[0,0].plot(effective_area_ratio_trend[12:], cp_theory_trend[12:,0], label=r'$C_{p,Nishino}$')
ax[0,0].scatter(effective_area_ratio, cp_statistical_model[:,0], s=30, marker='x', c='r', label=r'$C_{p,model}$')
ax[0,0].scatter(effective_area_ratio, cp_finite[:,0], s=30, marker='^', facecolors='none', edgecolors='k', label=r'$C_{p,LES}$')
ax[0,0].set_xlabel(r'$\lambda/C_{f0}$')
ax[0,0].set_ylabel(r'$C_p$')
ax[0,0].set_title(r'a) $\zeta=0$', loc='left')

ax[0,1].plot(effective_area_ratio_trend[12:], cp_theory_trend[12:,1])
ax[0,1].scatter(effective_area_ratio, cp_statistical_model[:,1], s=30, marker='x', c='r')
ax[0,1].scatter(effective_area_ratio, cp_finite[:,1], s=30, marker='^', facecolors='none', edgecolors='k')
ax[0,1].set_xlabel(r'$\lambda/C_{f0}$')
ax[0,1].set_ylabel(r'$C_p$')
ax[0,1].set_title(r'b) $\zeta=5$', loc='left')

ax[1,0].plot(effective_area_ratio_trend[12:], cp_theory_trend[12:,2])
ax[1,0].scatter(effective_area_ratio, cp_statistical_model[:,2], s=30, marker='x', c='r')
ax[1,0].scatter(effective_area_ratio, cp_finite[:,2], s=30, marker='^', facecolors='none', edgecolors='k')
ax[1,0].set_xlabel(r'$\lambda/C_{f0}$')
ax[1,0].set_ylabel(r'$C_p$')
ax[1,0].set_title(r'c) $\zeta=10$', loc='left')

ax[1,1].plot(effective_area_ratio_trend[12:], cp_theory_trend[12:,3])
ax[1,1].scatter(effective_area_ratio, cp_statistical_model[:,3], s=30, marker='x', c='r')
ax[1,1].scatter(effective_area_ratio, cp_finite[:,3], s=30, marker='^', facecolors='none', edgecolors='k')
ax[1,1].set_xlabel(r'$\lambda/C_{f0}$')
ax[1,1].set_ylabel(r'$C_p$')
ax[1,1].set_title(r'd) $\zeta=15$', loc='left')

ax[2,0].plot(effective_area_ratio_trend[12:], cp_theory_trend[12:,4])
ax[2,0].scatter(effective_area_ratio, cp_statistical_model[:,4], s=30, marker='x', c='r')
ax[2,0].scatter(effective_area_ratio, cp_finite[:,4], s=30, marker='^', facecolors='none', edgecolors='k')
ax[2,0].set_xlabel(r'$\lambda/C_{f0}$')
ax[2,0].set_ylabel(r'$C_p$')
ax[2,0].set_title(r'e) $\zeta=20$', loc='left')

ax[2,1].plot(effective_area_ratio_trend[12:], cp_theory_trend[12:,5])
ax[2,1].scatter(effective_area_ratio, cp_statistical_model[:,5], s=30, marker='x', c='r')
ax[2,1].scatter(effective_area_ratio, cp_finite[:,5], s=30, marker='^', facecolors='none', edgecolors='k')
ax[2,1].set_xlabel(r'$\lambda/C_{f0}$')
ax[2,1].set_ylabel(r'$C_p$')
ax[2,1].set_title(r'f) $\zeta=25$', loc='left')

fig.legend(bbox_to_anchor=(0.5,-0.05), loc="lower center", 
                bbox_transform=fig.transFigure, ncol=3)

plt.tight_layout()

plt.savefig('figures/cp_predictions.png', bbox_inches='tight')