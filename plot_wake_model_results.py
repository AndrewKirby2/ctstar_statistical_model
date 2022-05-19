import numpy as np
import matplotlib.pyplot as plt

#load wake model results
ctstar_wake_model = np.genfromtxt('ctstar_wake_model.csv', delimiter=',')
#load LES results
LES_results = np.genfromtxt('training_data.csv', delimiter=',')
ctstar_les = LES_results[:,3]

ti = [1, 5, 10, 15]

plt.figure()
for i in range(4):
    plt.scatter(ctstar_les, ctstar_wake_model[:,i], label='TI = '+str(ti[i])+'%')
plt.legend()

plt.ylim([0.45,0.85])
plt.xlim([0.55,0.8])
plt.plot([0.55,0.8],[0.55,0.8])
plt.xlabel(r'$C_{T,LES}^*$')
plt.ylabel(r'$C_{T,wake model}^*$')
plt.savefig('wake_model_results.png')