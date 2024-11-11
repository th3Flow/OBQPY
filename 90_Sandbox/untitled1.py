import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, tf2zpk
import scipy.signal as sigP
import cvxpy as cp

plt.close('all')

# Frequency range
v_w = np.arange(0, np.pi, step=0.01).reshape(-1, 1)

# Load an existing impulse response (IR) for the "desired" system
v_h_ls_input = np.load('vWcoeff.npy')
s_M = int(len(v_h_ls_input)*0.8) 

# Adjust delay parameter
s_Delay = s_M / 2  # You can adjust this value as needed

# Compute desired frequency response (with delay if needed)
v_H_ls_input = freqz(v_h_ls_input, 1, v_w.flatten())[1]
v_Dd = v_H_ls_input.flatten()  # * np.exp(-1j * v_w.flatten() * s_Delay)

# Visualize the desired magnitude and phase
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(v_w / np.pi, 20 * np.log10(np.abs(v_Dd)))
plt.title('Desired Magnitude (User-Provided IR)')
plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(v_w / np.pi, np.unwrap(np.angle(v_Dd)))
plt.title('Desired Phase (User-Provided IR)')
plt.grid(True)
plt.tight_layout()
plt.show()

# DTFT Matrix for the FIR Filter
v_n = np.arange(0, s_M + 1)
v_wd = np.arange(0, np.pi, step=0.01).reshape(-1, 1)
v_H_ls_input = freqz(v_h_ls_input, 1, v_wd.flatten())[1]
v_Dd = v_H_ls_input.flatten()  # * np.exp(-1j * v_w.flatten() * s_Delay)

m_F = np.exp(-1j * v_wd * v_n)

# Minimax design using convex optimization
v_h_mm = cp.Variable(s_M + 1)
objective_mm = cp.Minimize(cp.max(cp.abs(m_F @ v_h_mm - v_Dd)))
problem_mm = cp.Problem(objective_mm)
problem_mm.solve()

# Least-squares design
v_h_ls = cp.Variable(s_M + 1)
objective_ls = cp.Minimize(cp.norm(m_F @ v_h_ls - v_Dd, 2))
problem_ls = cp.Problem(objective_ls)
problem_ls.solve()

# Frequency response of minimax and least-squares filters
v_h_mm = sigP.minimum_phase(v_h_mm.value, n_fft = len(v_h_mm.value)*2, method='homomorphic', half = False)
v_H_mm = freqz(v_h_mm, 1, v_wd.flatten())[1]
v_h_ls = sigP.minimum_phase(v_h_ls.value, n_fft = len(v_h_ls.value)*2, method='homomorphic', half = False)
v_H_ls = freqz(v_h_ls, 1, v_wd.flatten())[1]

# Plot the frequency response and zero-pole diagram for Minimax design
plt.figure(figsize=(12, 10))

# Magnitude Response
plt.subplot(3, 1, 1)
plt.plot(v_wd / np.pi, 20 * np.log10(np.abs(v_Dd)), '--', label='Desired')
plt.plot(v_wd / np.pi, 20 * np.log10(np.abs(v_H_mm)), label='Minimax approximation')
plt.title('Magnitude Response (Minimax Design)')
plt.legend()
plt.grid(True)

# Phase Response
plt.subplot(3, 1, 2)
plt.plot(v_wd / np.pi, np.unwrap(np.angle(v_Dd)), '--', label='Desired')
plt.plot(v_wd / np.pi, np.unwrap(np.angle(v_H_mm)), label='Minimax approximation')
plt.title('Phase Response (Minimax Design)')
plt.legend()
plt.grid(True)

# Zero-Pole Diagram
plt.subplot(3, 1, 3)
z_mm, p_mm, _ = tf2zpk(v_h_mm, [1])
plt.plot(np.real(z_mm), np.imag(z_mm), 'o', label='Zeros')
plt.plot(np.real(p_mm), np.imag(p_mm), 'x', label='Poles')
# Unit circle for reference
theta = np.linspace(0, 2 * np.pi, 512)
plt.plot(np.cos(theta), np.sin(theta), 'k--', label='Unit Circle')
plt.title('Zero-Pole Diagram (Minimax Design)')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.grid(True)
plt.axis('equal')
plt.legend()

plt.tight_layout()
plt.show()

# Plot the frequency response and zero-pole diagram for Least-Squares design
plt.figure(figsize=(12, 10))

# Magnitude Response
plt.subplot(3, 1, 1)
plt.plot(v_wd / np.pi, 20 * np.log10(np.abs(v_Dd)), '--', label='Desired')
plt.plot(v_wd / np.pi, 20 * np.log10(np.abs(v_H_ls)), label='Least-Squares approximation')
plt.title('Magnitude Response (Least-Squares Design)')
plt.legend()
plt.grid(True)

# Phase Response
plt.subplot(3, 1, 2)
plt.plot(v_wd / np.pi, np.unwrap(np.angle(v_Dd)), '--', label='Desired')
plt.plot(v_wd / np.pi, np.unwrap(np.angle(v_H_ls)), label='Least-Squares approximation')
plt.title('Phase Response (Least-Squares Design)')
plt.legend()
plt.grid(True)

# Zero-Pole Diagram
plt.subplot(3, 1, 3)
z_ls, p_ls, _ = tf2zpk(v_h_ls, [1])
plt.plot(np.real(z_ls), np.imag(z_ls), 'o', label='Zeros')
plt.plot(np.real(p_ls), np.imag(p_ls), 'x', label='Poles')
# Unit circle for reference
plt.plot(np.cos(theta), np.sin(theta), 'k--', label='Unit Circle')
plt.title('Zero-Pole Diagram (Least-Squares Design)')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.grid(True)
plt.axis('equal')
plt.legend()

plt.tight_layout()
plt.show()
