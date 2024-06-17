This GitHub repository is to submit the used python files in my BEP project.
Before running any of the files, the following is needed:
GLENNI package from L. Hillmann
classGLE.py from https://zenodo.org/records/6607600 
which is the implementation of:
  Clemens Willers and Oliver Kamps
	Efficient Bayesian estimation of the generalized Langevin equation from data
	arXiv preprint arXiv:2107.04560v2, 2022
	https://doi.org/10.48550/arXiv.2107.04560
 Some files are used for both the real-world data as well as the artificial data. 
 The real-world data can be downloaded from https://data.open-power-system-data.org/time_series/2020-10-06 ('time_series_15min_singleindex.csv'), additional the used array for solar power generation in Austria is provided in repository under 'data15min.npy' 
 The artificial data is both directly uploaded to this repository (under artificialdata.npy) or is computed in different files such as 'ARIMA_fitting.py'

 The files can best be used in the following fashion:
 1. ck_plot.py, define the dataset (array) used under dataset = np.load('downloads/data15min.npy') the plot that is generated can be used to determine the length of the memory kernel K.
 2. main.py, define the dataset used under dataset = np.load('downloads/data15min.npy') and set memory length model.setMemoryLength(728), this file generates and saves an np.array that contains the weight of the kernel.
 3. bayesian_memory_kernel_plot.py, Opens the Bayesian memorykernel and generate the plot, memory_kernel_values = np.load('downloads/real48.npy').
 4. fitting_AP_from_Bayesiankernel.py, also uses the Bayesian memory kernel to generate an A_p. Furthermore, it creates a plot to compare the original input data and the generated data from the computed A_p.
 5. ARIMA_fitting.py, takes the orginal data  set from https://data.open-power-system-data.org/time_series/2020-10-06 or computes the artificial data and creates a plot with a year's worth of an ARIMA simulated trajectory.
 6. Bayesiankernel_for_same_AP.py, this file takes two A_p and generates a plot with 4 subplots to show if the calculated Bayesian memory kernel is always the same for the same A_P, but different realizations.
 7. kernels_comparison.py, this gives a plot where the memory kernel of the Bayesian memory kernel is compared with the exact memory kernel directly from A_p.
    
