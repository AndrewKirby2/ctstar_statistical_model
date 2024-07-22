# ctstar_statistical_model
Clone remote GitHub repository
```
git clone https://github.com/AndrewKirby2/ctstar_statistcal_model.git
```
Change working directory
```
cd ctstar_statistical_model
```
Create python virtual environment
```
python3 -m venv venv
```
Activate virtual environment
```
source venv/bin/activate
```
Install dependencies
```
pip install -r requirements.txt
```
The code for the low-order wake model (see section 3.2) is in the script
```
low_order_wake_model.py
```
Perform LOOCV for standard GP models (see section 5.1). This uses data from Large-Eddy Simulations (data/LES_training_data.csv) and from the low-order wake model (data/ctstar_wake_model.csv)
```
python standard_GP_LOOCV.py
```
Plot the posterior variance for the GP-wake-TI10-prior (figure is saved into the folder "figures")
```
python GP-wake-TI10-prior_posterior_variance.py
```
Perform LOOCV for the multi-fidelity GP models (see section 5.2). This uses data from Large-Eddy Simulations (data/LES_training_data.csv) and low-order wake models (data/wake_model_maximin_*)
```
python mf_GP_LOOCV.py
```
Plot the posterior distrubtions from MF-GP-nlow500 (i.e., figures 8, 9 and 11 in Kirby et. al. 2023)
```
python MF-GP-nlow500_posterior.py
```
Use predictions of C_T^* from MF-GP-nlow500 to predict farm power coefficient and compare with LES results and C_{p,Nishino}
```
python cp_prediction_LOOCV.py
```
Predict C_T^* using MF-GP-nlow500 for S_x = 7.5, S_y = 5.0, theta = 0 (deg)
```
python
from trained_models.MF_GP_nlow500 import *
MF_GP_nlow500(7.5, 5.0, 0.0)
```
Note that the prediction and predictive variance are returned.

The LES code for the wind farm actuator disk simulations can be found at https://code.metoffice.gov.uk/svn/monc/main/branches/dev/andrewkirby/r8169_turbines_v3 with the turbine implementation at /components/turbines/src/turbines.F90 (note a Met Office MOSRS account is required to assess this).
This code will produce incorrect results if:
- the grid resolution is so high that >10,000 grid points lie within a sphere with a diameter equal to turbine diameters which is centred on the centre of a turbine disk
- any turbine centre is within 1 turbine diameter of a boundary (e.g. top, bottom or side)
- any turbine centre is less than 2 turbine diameters away from another turbine centre.

