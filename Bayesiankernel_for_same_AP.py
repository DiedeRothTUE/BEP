import numpy as np
import glenni.ceriotti as ceriotti
import matplotlib.pyplot as plt
from classGLE import GLE

# Define two simple Ap matrices
A_P1 = np.array([[0.0, 2], [-2, 0]])
A_P2 = np.array([[0.0, 16], [-16, 0.05]])

# Parameters for the simulation
dt = 0.1
n_steps = 200
n_output = 1

# Function to generate positions using GLENNI
def generate_positions(A, dt, n_steps, n_output):
    system = ceriotti.initialise(A, 1)
    positions, velocities = ceriotti.integrate(system, A, dt, n_steps, n_output, 1)
    positions = positions[:, 0]
    return positions

# Initialize the GLE model
model = GLE(dt=dt)

# Memory length and bins
memory_length = 20
bins = 1

# Function to process the dataset and compute the Bayesian memory kernel
def process_positions(positions, tau=1):
    model.readData(positions, tau=tau)
    model.setMemoryLength(memory_length)
    model.calcBins(bins)
    model.calcPlotKappa()
    model.directEstimation_SLE()
    model.calcCoefficients()
    model.MAPestimation(printOptSuccess=True)
    return model.theta_MAPest[str(model.K)][2 * model.N_bins:]

# Generate three realizations for each Ap and calculate the memory kernels
memory_kernels_Ap1 = []
memory_kernels_Ap2 = []

for _ in range(3):
    positions_Ap1 = generate_positions(A_P1, dt, n_steps, n_output)
    memory_kernel_Ap1 = process_positions(positions_Ap1)
    memory_kernels_Ap1.append(memory_kernel_Ap1)
    
    positions_Ap2 = generate_positions(A_P2, dt, n_steps, n_output)
    memory_kernel_Ap2 = process_positions(positions_Ap2)
    memory_kernels_Ap2.append(memory_kernel_Ap2)

# Function to plot memory kernels

fig, axs = plt.subplots(1, 2, figsize=(15, 7))

k_values = np.arange(0, memory_length, 1)

# Plot for Ap1
for kernel in memory_kernels_Ap1:
    axs[0].plot(k_values, kernel, linestyle='-', marker='o', markersize=4, linewidth=1)
axs[0].set_title('Memory Kernel Values for Ap1', fontsize=16)
axs[0].set_xlabel('Lagged Value (k)', fontsize=14)
axs[0].set_ylabel('Weight ($K_k$)', fontsize=14)
axs[0].grid(True)
axs[0].tick_params(axis='both', labelsize=12)


for kernel in memory_kernels_Ap2:
    axs[1].plot(k_values, kernel, linestyle='-', marker='o', markersize=4, linewidth=1)
axs[1].set_title('Memory Kernel Values for Ap2', fontsize=16)
axs[1].set_xlabel('Lagged Value (k)', fontsize=14)
axs[1].set_ylabel('Weight ($K_k$)', fontsize=14)
axs[1].grid(True)
axs[1].tick_params(axis='both', labelsize=12)
plt.tight_layout()


plt.savefig('downloads/plotsthesis/memory_kernelssame.png')


plt.show()
