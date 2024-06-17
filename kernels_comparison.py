import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from numpy.linalg import eig
import math
import glenni.ceriotti as ceriotti
from classGLE import GLE

# the memory kernel obtained directly from A_p (if A_p is of the form (11) presented in the paper)
def memory_kernel_li(t, c1, c2, a, b):
    memory = []
    lambda1 = c1**2 + c2**2
    lambda2 = (c2**2 - c1**2) * (a / (4 * b**2 - a**2)**0.5)
    w = (4 * b**2 - a**2) / 2
    for x in t:
        result = np.exp(-(a / 2) * x) * (lambda1 * math.cos(w * x) + lambda2 * math.sin(w * x))
        memory.append(result)
    return memory

# Function to generate positions and velocities
def generate_positions(A, dt, n_steps, n_output):
    system = ceriotti.initialise(A, 1)
    positions, velocities = ceriotti.integrate(system, A, dt, n_steps, n_output, 1)
    positions = positions[:, 0]
    return positions

# Function to perform Bayesian approximation
def bayesian_memory_kernel(positions, tau, memory_length, bins):
    model = GLE(dt=0.1)
    model.readData(positions, tau=tau)
    model.setMemoryLength(memory_length)
    model.calcBins(bins)
    model.calcPlotKappa()
    model.directEstimation_SLE()
    model.calcCoefficients()
    model.MAPestimation(printOptSuccess=True)
    return model.theta_MAPest[str(model.K)][2 * model.N_bins:]

# Parameters for the two A matrices
c11= 1
c21= 1
a1 = 1
b1= 1/4
c12= -4.956925318
c22= 7.711205539451
a2= 2.4981
b2 = 2.849899
A1 = np.array([[0, c11, c21], [-c11, a1, b1], [-c21, -b1, 0]], dtype=float)
A2 = np.array([[0, c12, c21], [-c12, a2, b2], [-c22, -b2, 0]], dtype=float)

# Common parameters
dt = 0.1
n_steps = 1000
n_output = 1
tau = 1
memory_length = 200
bins = 1

# Generate positions for both matrices
positions1 = generate_positions(A1, dt, n_steps, n_output)
positions2 = generate_positions(A2, dt, n_steps, n_output)

# Calculate memory kernels using Bayesian approximation
memory_bayesian1 = bayesian_memory_kernel(positions1, tau, memory_length, bins)
memory_bayesian2 = bayesian_memory_kernel(positions2, tau, memory_length, bins)

#timeline accounting for the timestep
tt = dt * np.arange(0, memory_length, 1)

# Calculate memory kernels using Li's method
memory_li1 = memory_kernel_li(tt, c11, c12, a1, b1)
memory_li2 = memory_kernel_li(tt, c12, c22, a2, b2)

# Plotting all results in a 2x2 subplot
plt.figure(figsize=(14, 10))
plt.title('Memory kernels for different A_p')
# Plot memory kernel for A1 using Bayesian approximation
plt.subplot(2, 2, 1)
plt.plot(tt, memory_bayesian1,)
plt.title('Memory Kernel (Bayesian) for A_p 1')
plt.xlabel('Time')
plt.ylabel('Memory')
plt.legend()


plt.subplot(2, 2, 2)
plt.plot(tt, memory_li1, color='red')
plt.title('Memory Kernel constructed from A_p 1')
plt.xlabel('Time')
plt.ylabel('Memory')
plt.legend()


plt.subplot(2, 2, 3)
plt.plot(tt, memory_bayesian2, label='Bayesian Approximation')
plt.title('Memory Kernel (Bayesian) for A_p 2')
plt.xlabel('Time')
plt.ylabel('Memory')
plt.legend()


plt.subplot(2, 2, 4)
plt.plot(tt, memory_li2, color='red')
plt.title('Memory Kernel constructed from A_p 2')
plt.xlabel('Time')
plt.ylabel('Memory')
plt.legend()

plt.tight_layout()
plt.savefig('downloads/plotsthesis/memory_kernels_comparison.png')
plt.show()
