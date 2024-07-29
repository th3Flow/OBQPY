import numpy as np

# Original input signal
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# Original filter coefficients
h = np.array([0.25, 0.5, 0.75, 1.0])

# Delay length
d = 2

# Split the filter into delay and non-causal parts
h_delay = h[:d]
h_non_causal = h[d:]

# Function to apply delay component
def apply_delay(x, h_delay):
    N = len(x)
    M = len(h_delay)
    y = np.zeros(N)
    for n in range(N):
        for k in range(M):
            if n - k >= 0:
                y[n] += h_delay[k] * x[n - k]
    return y

# Function to apply non-causal component
def apply_non_causal(x, h_non_causal, d):
    N = len(x)
    M = len(h_non_causal)
    y = np.zeros(N)
    for n in range(N):
        for k in range(M):
            if n - k - d >= 0:
                y[n] += h_non_causal[k] * x[n - k - d]
    return y

# Apply delay component
y_delay = apply_delay(x, h_delay)

# Apply non-causal component
y_non_causal = apply_non_causal(x, h_non_causal, d)

# Combine the results
y_combined = y_delay + y_non_causal

# Print results
print("Original Input Signal:", x)
print("Delay Filter Coefficients:", h_delay)
print("Non-Causal Filter Coefficients:", h_non_causal)
print("Output after Delay Component:", y_delay)
print("Output after Non-Causal Component:", y_non_causal)
print("Combined Output:", y_combined)