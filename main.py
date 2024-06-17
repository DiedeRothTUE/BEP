import numpy as np
from classGLE import *
#from classcerrioti import *
import glenni.ceriotti as ceriotti
import matplotlib.pyplot as plt
#this main is obtained from https://zenodo.org/records/6607600 and then adjusted
model = GLE(dt = 0.1)

# Load your dataset
dataset = np.load('downloads/data15min.npy')  #load the data
model.readData(dataset, tau = 48) #tau indicates how many observations are skipped

model.setMemoryLength(728)

model.calcBins(10) #the amount of bins the data is seperated in

model.calcPlotKappa()

model.directEstimation_SLE()

model.plotD12_SLEdirect()

model.calcCoefficients()

model.MAPestimation(printOptSuccess = True)

model.plotD12x(withDirectEstimation = True)
model.plotKernel()

np.save('downloads/real48.npy', model.theta_MAPest[str(model.K)][2 * model.N_bins:]) #saving the array that is the memory kernel in a seperate file
