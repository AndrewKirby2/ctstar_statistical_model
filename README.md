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
Perform LOOCV for standard GP model (see section 5.1). This uses data from Large-Eddy Simulations (data/LES_training_data.csv) and from the low-order wake model (data/ctstar_wake_model.csv)
```
python standard_GP_LOOCV.py
```
Plot the posterior variance for the \textbf{GP-wake-TI10-prior} (figure is saved into the folder "figures")
```
python GP-wake-TI10-prior_posterior_variance.py
```
