import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Parameters
fs = 4000  # Sampling frequency (lower to enhance aliasing effect)
f_signal = 125  # Modulating signal frequency
f_pwm = 300  # Higher PWM carrier frequency to induce aliasing
t = np.linspace(0, 1, fs, endpoint=False)  # Time vector

# Generate a low-frequency sine wave as the modulating signal
signal = 0.5 * (1 + np.sin(2 * np.pi * f_signal * t))

# Generate PWM signal
pwm = np.zeros_like(signal)
for i in range(len(signal)):
    pwm[i] = 1 if signal[i] > (i % int(fs/f_pwm)) / (fs/f_pwm) else 0

# Calculate FFT of PWM signal
fft_pwm = fft(pwm)
fft_freqs = fftfreq(len(pwm), 1 / fs)

# Plot the magnitude spectrum of the PWM signal to demonstrate aliasing
plt.figure(figsize=(10, 6))
plt.plot(fft_freqs[:fs//2], np.abs(fft_pwm[:fs//2]) / len(pwm), label="PWM Magnitude Spectrum")
plt.axvline(fs/2, color='r', linestyle='--', label='Nyquist Frequency')
plt.title("Magnitude Spectrum of PWM Signal (Enhanced Aliasing Demonstration)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.legend()
plt.grid()
plt.show()
