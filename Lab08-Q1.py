# Computing the temperature distrubution in a heat conductor
# Author: Nico Grisouard, University of Toronto
# Heavily inspired by Newman's laplace.py
# Date: 24 October 2018

import numpy as np
import matplotlib.pyplot as plt

# Constants
M = 100  # Grid squares on a side
V = 1.0  # Voltage at top wall
omega = 0.9  # overrelaxation parameter
target = 1e-6   # Target accuracy
ite_max = 1e9

Lx = 20e-2  # [m] x-length of domain
Ly = 8e-2  # [m] y-length of domain
AB = 5e-2  # [m] length of AB
CD = 10e-2  # [m] length of CD
BC = 3e-2  # [m] length of BC
a = 0.1e-2  # [m] dx

anim = False  # will priduce animation if True; will print final figure if not

x = np.arange(0, Lx+a, a)  # +a otherwise the last point is not Lx
y = np.arange(0, Ly+a, a)
Mx = len(x)
My = len(y)
X, Y = np.meshgrid(x, y)  # for plotting

# below, I assume that missing the indices by +/-1 will be OK
iB = int(AB/a)  # x-index of where B and C are located
iD = int((AB + CD)/a)  # x-index of where D and E are located
jC = int((Ly - BC)/a)  # y-index of where C and D are located

# Create arrays to hold temperature values
T = np.zeros([My, Mx], float)

# Boundary conditions for the temperature
T[-1, :iB] = 5*x[:iB]/AB  # along AB (in that order)
T[jC:, iB] = 2.*(Ly - y[jC:])/BC + 5.  # along CB (i.t.o.)
T[jC, iB:iD] = 7.  # along CD (i.t.o.)
T[jC:, iD] = 2.*(Ly - y[jC:])/BC + 5.  # along DE (i.t.o.)
T[-1, iD:] = 5.*(Lx - x[iD:])/AB  # along EF (i.t.o.)
T[:, -1] = 10*(1-y/Ly)  # along FG (in that order)
T[0, :] = 10.  # along HG (in that order)
T[:, 0] = 10*(1-y/Ly)  # along HA (in that order)


# Create mask to know which cells are being computed
mask = np.ones([My, Mx], bool)
mask[:, 0] = 0  # We will leave those points intact
mask[0, :] = 0  # idem
mask[:, -1] = 0
mask[-1, :] = 0
mask[jC:-1, iB:iD+1] = 0


# Main loop
delta = 1.0
T_old = 1*T
it = 1  # iterations counter
while delta > target and it < ite_max:
    # Calculate new values of the potential
    for j in range(1, My-1):
        for i in range(1, Mx-1):
            if mask[j, i]:
                T[j, i] = (T[j+1, i] + T[j-1, i] +
                           T[j, i+1] + T[j, i-1])*(1+omega)*0.25 \
                    - omega*T[j, i]

    # Calculate maximum difference from old values
    delta = np.max(abs(T-T_old))
    T_old = 1*T  # point of comparison for next convergence measurement
    it += 1

    if anim:
        plt.clf()
        # Make a plot
        plt.contourf(X, Y, T, 32)   # 32 filled contours
        plt.xlabel('$x$ (m)')
        plt.xlabel('$y$ (m)')
        plt.colorbar(orientation='horizontal')
        plt.draw()
        plt.pause(0.01)


# Temperature at the desired location
ix = int(2.5e-2/a)
jy = int(1.e-2/a)
print("At x = 2.5 cm, y = 1 cm: T=", T[jy, ix])
print("At x = 17.5 cm, y = 1 cm: T=", T[jy, -ix-1])
print("At x = 2.5 cm, y = 7 cm: T=", T[-jy-1, ix])
print("At x = 17.5 cm, y = 7 cm: T=", T[-jy-1, -ix-1])

if not anim:
    plt.contourf(X, Y, T, 32)   # 32 filled contours
    plt.xlabel('$x$ (m)')
    plt.xlabel('$y$ (m)')
    plt.colorbar(orientation='horizontal')
    plt.savefig('Lab-Q1-T-om{0}-it{1}.png'.format(int(omega*10), it))
    plt.show()
