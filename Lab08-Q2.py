# Computing the vibrations of a string, hit by a hammer
# Author: Nico Grisouard, University of Toronto
# Inspired by Newman's heat.py
# Date: 24 October 2018

import numpy as np
import matplotlib.pyplot as plt

# %% Adjustable parameters ---------------------------------------------------|
L = 1.  # [] Length of string
v = 100.  # [m/s] wave propagation speed
d = 10.e-2  # [m] location where string is hit
C = 1.  # [m/s] transverse velocity amplitude
sig = 0.3  # [m] typical width of initial disturbance
h = 1e-6  # [s] time step
a = 1.e-2  # [m] Grid step

# Times at which I print a snapshot
t_record = [5.5e-2, 7.5e-2, 9.5e-2, .1]
epsilon = h*0.5  # tolerance to detect when t_record is hit


# %% Dependent parameters ----------------------------------------------------|
x = np.arange(0, L+a, a)
N = len(x)

# Create arrays
phi = np.zeros(N, float)
psi = C*x*(L-x)/L**2 * np.exp(-0.5*(x-d)**2/sig**2)
phi_record = np.empty((4, N), float)  # For a still figure for the solutions

phip = np.zeros(N, float)
psip = np.empty(N, float)

# %% Main loop ---------------------------------------------------------------|
t = 0.0
ii = 0  # subplot counter
while max(abs(phi)) < L and t < .1:

    # Calculate the new values of phi
    for i in range(1, N-1):
        psip[i] = psi[i] + h*v**2/a**2*(phi[i+1] + phi[i-1] - 2*phi[i])
        phip[i] = phi[i] + h*psi[i]
    phi, phip, psi, psip = phip, phi, psip, psi
    t += h

    # Make plots at the given times
    if abs(t-t_record[ii]) < epsilon:
        phi_record[ii, :] = phi
        ii += 1

    if int(t/h) % 100 == 0:
        # continue  # comment to see the animation
        plt.figure(1)
        plt.clf()
        # Make a plot
        plt.plot(x, phi)   # 32 filled contours
        plt.xlabel('$x$ (m)')
        plt.ylabel('$y$ (m)')
        plt.xlim([0., L])
        plt.ylim([-5e-4, 5e-4])
        plt.title('$t = ${0:.4f} s'.format(t))
        plt.grid()
        plt.draw()
        plt.pause(0.01)


# %% Last plot ---------------------------------------------------------------|
plt.figure(2)
for ii in range(4):
    plt.subplot(2, 2, ii+1)
    plt.plot(x, phi_record[ii, :]*1000)
    plt.xlabel("$x$ (m)")
    plt.ylabel("$y$ (mm)")
    plt.xlim([0., L])
    plt.ylim([-5e-1, 5e-1])
    plt.title("$y$ at $t=${0:.4} s".format(t_record[ii]))
    plt.grid()

plt.tight_layout()
plt.savefig('Lab08-Q2.png')
plt.show()
