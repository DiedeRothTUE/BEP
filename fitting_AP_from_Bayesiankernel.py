from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import glenni.ceriotti as ceriotti

# Load real memory kernel 
memory_real = np.load('downloads/perfectmemorykernel.npy').astype(np.float64)
# Make sure that the memory length is equal to tt
tt = np.arange(0, 100, 1).astype(np.float64)

# This function fits all the parameters, to the expontially damped cosine functions
def memory_kernel_form(t, *params):
    n = len(params) // 4  # Number of terms (dividing by 4 because 4 parameters per temr )
    kernel = np.zeros_like(t, dtype=np.float64)  #kernel represents the fitting terms 
    for i in range(n):
        p = params[4 * i]
        q = params[4 * i + 1]
        r = params[4 * i + 2]
        s = params[4 * i + 3]
        kernel += p * np.exp(-q * t) * np.cos(r * t - s).astype(np.float64)
    return kernel


#this function gives some random guesses for the parameter terms, given the amount of paremrters
def generate_initial_guesses(num_terms):
    initial_guesses = []
    for i in range(num_terms):
        initial_guesses += [1.0, 0.01, 0.1, 0.0]  # Example initial guess, as floats
    return initial_guesses


terms_list = [20] # the numbers in this list determine the different number of terms that are fitted
fits = {}

for num_terms in terms_list:
    initial_guesses = generate_initial_guesses(num_terms)
    #here we fit, given the number of terms the parameters to the bayesian kernel
    popt, _ = curve_fit(memory_kernel_form, tt, memory_real, p0=initial_guesses, maxfev=900000)
    fitted_params = popt
    fits[num_terms] = memory_kernel_form(tt, *fitted_params).astype(np.float64)

# tThe original and fitted memory kernels
plt.figure(figsize=(10, 6))
plt.plot(tt, memory_real, label='Original Memory Kernel', linewidth=0.5)
for num_terms, kernel_values in fits.items():
    plt.plot(tt, kernel_values, label=f'Fitted Memory Kernel ({num_terms} terms)')

plt.xlabel('Time')
plt.ylabel('Memory Kernel Value')
plt.title('Memory Kernel Plot with Different Numbers of Terms')
plt.legend()
plt.grid(True)
plt.savefig('downloads/fittedkernel_comparison.png', dpi=900)

# Plot residuals for each fitting
plt.figure(figsize=(10, 6))
for num_terms, kernel_values in fits.items():
    residuals = memory_real - kernel_values
    print(sum(residuals))
    plt.plot(tt, residuals, label=f'Residuals ({num_terms} terms)')

plt.xlabel('Time')
plt.ylabel('Residual Value')
plt.title('Residuals Plot for Different Numbers of Terms')
plt.legend()
plt.grid(True)
plt.savefig('downloads/residuals_comparison.png', dpi=900)

plt.show()
print((8.009210368087327e-06 - -3.110946808376758e-05))
#n is here two times the number of terms that are fitted 
n = 40
#Defining the A_p matrix (add an extra entry for the 0 on [1,1])
A = np.zeros((n + 1, n + 1), dtype=float)
#the matrix will now be filled with parameters obtained form the last curvefit (thus also with number of terms that was last in terms_list)
params = fitted_params.reshape(20, 4)
for x in range(0, 40, 2):
    t = int(x / 2)
    
    row = params[t]  # Extracting the correct slice of parameters
    p = row[0]
    q = row[1]
    r = row[2]
    s = row[3]

    try: #here the values of the parameters in A_p are obtained, due to possibility of negative number under sqrt, the c1 and c2. The absolute value is then taken.
        a = 2 * q
        b = np.sqrt(r**2 + q**2)
        c=  abs((p / 2) * np.cos(s) - (r * p / (2 * q)) * np.sin(s))
        c1 = np.sqrt(c)
        c = abs((p / 2) * np.cos(s) + (r * p / (2 * q)) * np.sin(s))
        c2 = np.sqrt(c)
        
        
        #this fills the matrix
        A[0][x + 1] = c1
        A[0][x + 2] = c2
        A[x + 1][0] = -c1
        A[x + 2][0] = -c2
        A[x + 1][x + 1] = a
        A[x + 1][x + 2] = b
        A[x + 2][x + 1] = -b
    
    except RuntimeWarning:
        print(a,b,c1,c2)
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
print(A)
#Generating, given the obtained A positions from the GLENNI package
dt = 0.001
n_steps = 8000
n_output = 1
system = ceriotti.initialise(A, 1)
positions, velocities = ceriotti.integrate(system, A, dt, n_steps, n_output, 1)
positions = positions[:, 0]
tt = dt * np.arange(0, n_steps, n_output)

combined_signal = np.load('downloads/artificial_data')
plt.figure(figsize=(12, 8))

# Plot positions over time
plt.subplot(2, 1, 1)
plt.plot(tt, positions)
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('GLE generated positions over Time')
plt.legend()
plt.grid(True)

# Plot combined signal with periodic components and noise
plt.subplot(2, 1, 2)

plt.plot(t, combined_signal)
plt.xlabel('Time')
plt.ylabel('Signal Value')
plt.title('Artificial data with 2 sinus functions')
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig('downloads/combined_real2.png', dpi=900)