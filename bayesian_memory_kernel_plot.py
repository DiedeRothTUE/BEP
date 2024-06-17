import numpy as np
import matplotlib.pyplot as plt

#load the memory kernel, obtained from the bayesian approximation
memory_kernel_values = np.load('downloads/real48.npy')  
print(memory_kernel_values)
# Create the plot
plt.figure(figsize=(12, 6))
#change this to the actual length of the kernel
k_values = np.arange(0,728,1)  
plt.plot(k_values, memory_kernel_values, linestyle='-', marker='o', markersize=4, linewidth=1, color='blue')

# Adding labels and title
plt.xlabel('Lagged Value (k)', fontsize=14)
plt.ylabel('Weight ($K_k$)', fontsize=14)
plt.title('Memory Kernel Values', fontsize=16)
plt.grid(True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Save the plot as PNG
plt.savefig('downloads/plotsthesis/real728.png')

# Show the plot
plt.show()
