import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

# Filter order of FIR Filter
s_M = 20

# frequencies
v_w = np.arange(0, np.pi, 0.01).reshape(-1, 1)

# frequency response to approximate
v_G = 1 / (1 + 1j * v_w / (0.4 * np.pi))

# Plot desired magnitude and phase
plt.figure()
plt.subplot(211)
plt.plot(v_w / np.pi, 20 * np.log10(np.abs(v_G)))
plt.title('Desired Magnitude')
plt.grid(True)
plt.subplot(212)
plt.plot(v_w / np.pi, np.unwrap(np.angle(v_G)))
plt.title('Desired Phase')
plt.grid(True)
plt.show()

# Define bands of interest
s_K = 15 * s_M  # rule of thumb
v_wdp = np.linspace(0, 0.9 * np.pi, s_K).reshape(-1, 1)
v_wd = np.concatenate([-v_wdp[::-1], v_wdp])  # actually not necessary, since CVX search for a real solution

# Desired frequency response with additional Delay (linear phase)
s_Delay = s_M / 2
# s_Delay = 0  # try it
v_D = 1 / (1 + 1j * v_wd / (0.4 * np.pi)) * np.exp(-1j * v_wd * s_Delay)

# DTFT Matrix for the Filter; i.e., v_H = m_F*v_h
v_n = np.arange(0, s_M + 1).reshape(-1, 1)
m_F = np.exp(-1j * v_wd * v_n.T)

# Second-order cone programming using cvxpy
v_h_mm = cp.Variable(s_M + 1)
objective = cp.Minimize(cp.max(cp.abs(m_F @ v_h_mm - v_D)))
constraints = []
problem = cp.Problem(objective, constraints)
problem.solve()

v_h_ls = cp.Variable(s_M + 1)
objective_ls = cp.Minimize(cp.norm(m_F @ v_h_ls - v_D, 2))
problem_ls = cp.Problem(objective_ls, constraints)
problem_ls.solve()

# Frequency response for the designed filter
v_H_mm = np.fft.fft(v_h_mm.value, n=len(v_w))
v_Dd = 1 / (1 + 1j * v_w / (0.4 * np.pi)) * np.exp(-1j * v_w * s_Delay)
plt.figure()
plt.subplot(211)
plt.plot(v_w / np.pi, 20 * np.log10(np.abs(v_Dd)), '--', label='desired')
plt.plot(v_w / np.pi, 20 * np.log10(np.abs(v_H_mm)), label='LS approximation')
plt.title('Magnitude')
plt.legend(loc='SouthWest')
plt.grid(True)
plt.subplot(212)
plt.plot(v_w / np.pi, np.unwrap(np.angle(v_Dd)), '--', label='desired')
plt.plot(v_w / np.pi, np.unwrap(np.angle(v_H_mm)), label='LS approximation')
plt.title('Phase')
plt.legend(loc='SouthWest')
plt.grid(True)
plt.show()

v_Hd_mm = np.fft.fft(v_h_mm.value, n=len(v_wdp))
v_Dp = 1 / (1 + 1j * v_wdp / (0.4 * np.pi)) * np.exp(-1j * v_wdp * s_Delay)
plt.figure()
plt.plot(v_wdp / np.pi, 20 * np.log10(np.abs(v_Dp - v_Hd_mm)))
plt.title('Approximation Error |E(ω)|')
plt.grid(True)
plt.axis([0, 0.9, -100, 0])
plt.show()

v_Hd_ls = np.fft.fft(v_h_ls.value, n=len(v_wdp))
plt.figure()
plt.plot(v_wdp / np.pi, 20 * np.log10(np.abs(v_Dp - v_Hd_mm)), label='Minmax design')
plt.plot(v_wdp / np.pi, 20 * np.log10(np.abs(v_Dp - v_Hd_ls)), label='Least-squares design')
plt.title('Approximation Error |E(ω)|')
plt.grid(True)
plt.axis([0, 0.9, -100, 0])
plt.legend()
plt.show()