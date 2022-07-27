import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

#load LES results
LES_results = np.genfromtxt('training_data.csv', delimiter=',')

#load statistical model results
stat_model_mean = np.genfromtxt('ctstar_nonlin_statistical_model.csv', delimiter=',')
stat_model_std = np.genfromtxt('ctstar_nonlin_statistical_model_std.csv', delimiter=',')

standardised_errors = (stat_model_mean - LES_results[:,3])/stat_model_std

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=[8,6])
ax[0,0].scatter(stat_model_mean, standardised_errors)
ax[0,0].set_ylim([-3.5,3.5])
ax[0,0].axhline(y=0)
ax[0,0].set_xlabel(r'$C_T^*$ prediction')

ax[0,1].scatter(LES_results[:,0], standardised_errors)
ax[0,1].set_ylim([-3.5,3.5])
ax[0,1].set_xlabel(r'$S_x$')
ax[0,1].axhline(y=0)

ax[1,0].scatter(LES_results[:,1], standardised_errors)
ax[1,0].set_ylim([-3.5,3.5])
ax[1,0].set_xlabel(r'$S_y$')
ax[1,0].axhline(y=0)

ax[1,1].scatter(LES_results[:,2], standardised_errors)
ax[1,1].set_ylim([-3.5,3.5])
ax[1,1].set_xlabel(r'$\theta$ ($^o$)')
ax[1,1].axhline(y=0)

plt.tight_layout()
plt.savefig('gp_diagnostics_standardised.png')
plt.close()

errors = (stat_model_mean - LES_results[:,3])

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=[8,6])
ax[0,0].scatter(stat_model_mean, errors)
#ax[0,0].set_ylim([-3.5,3.5])
ax[0,0].axhline(y=0)
ax[0,0].set_xlabel(r'$C_T^*$ prediction')

ax[0,1].scatter(LES_results[:,0], errors)
#ax[0,1].set_ylim([-3.5,3.5])
ax[0,1].set_xlabel(r'$S_x$')
ax[0,1].axhline(y=0)

ax[1,0].scatter(LES_results[:,1], errors)
#ax[1,0].set_ylim([-3.5,3.5])
ax[1,0].set_xlabel(r'$S_y$')
ax[1,0].axhline(y=0)

ax[1,1].scatter(LES_results[:,2], errors)
#ax[1,1].set_ylim([-3.5,3.5])
ax[1,1].set_xlabel(r'$\theta$ ($^o$)')
ax[1,1].axhline(y=0)

plt.tight_layout()
plt.savefig('gp_diagnostics.png')

fig = sm.qqplot(standardised_errors, line='45')
plt.savefig('qq_plots.png')