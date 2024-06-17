import numpy as np
import matplotlib.pyplot as plt
#this functions calculates the value for a single bin
def calc_ck(data, K):
    N = len(data)
    ck = np.zeros(K)
    for k in range(1, K + 1):
        valid_indices = np.arange(k, N) 
        ck[k - 1] = np.sum(data[valid_indices] - data[valid_indices - k]) #it takes the sum for all valid indices (thus up to K) and creates the sum of differences
    return ck
#to detrend the data, every segment (being a year) is equal to 68 (mean of first year for simplicitiy)
def adjust_segment_mean(arr, segment_length=728, target_mean=68):
    n = len(arr)
    num_segments = n // segment_length
    adjusted_arr = np.copy(arr)
    
    for i in range(num_segments):
        segment = arr[i * segment_length: (i + 1) * segment_length]
        segment_mean = np.mean(segment)
        adjustment = target_mean - segment_mean
        adjusted_arr[i * segment_length: (i + 1) * segment_length] =  djusted_arr[i * segment_length: (i + 1) * segment_length] + adjustment
    

# the data, segment length is a year because there are 728 observations. 
dataset = np.load('downloads/data15min.npy') 
dataset = dataset[::48]
for x in range(0,4195,728):
    print(dataset[x-728:x].mean())
    dataset[x-728:x] = dataset[x-728:x]

dataset = adjust_segment_mean(dataset, segment_length=728, target_mean=68)

# Verify the means of each segment
for x in range(0, len(dataset), 728):
    segment = dataset[x:x + 728]
    print(segment.mean)

# Parameters
K = len(dataset) # Number of lags lengths to consider

# Calculate ck
ck = calc_ck(dataset, K)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(range(1, K + 1), ck)
plt.xlabel('lagged values K')
plt.ylabel('$c_k$')
plt.title('$c_k$ for the Dataset')
plt.grid(True)
plt.savefig('downloads/plotsthesis/ckplotrealreduced48.png',format = 'png')