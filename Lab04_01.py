# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 11:08:15 2018

@author: Heng
"""

import numpy as np
from numpy.linalg import solve
from SolveLinear import GaussElim, PartialPivot
from time import time
import matplotlib.pyplot as plt


'''
N = range(5,200)
time_GE = np.ones(len(N))
time_PP = np.ones(len(N))
time_LU = np.ones(len(N))
err_GE = np.ones(len(N))
err_PP = np.ones(len(N))
err_LU = np.ones(len(N))

for n in N:
    i = n-N[0]
    A = np.random.rand(n,n)
    v = np.random.rand(n)
    solutions = np.ones([3,n])

    
    # Gauss elimination
    start = time()
    solutions[0]=GaussElim(A,v)
    end = time()
    time_GE[i] = end - start
    v_sol = np.dot(A,solutions[0])
    err_GE[i] = np.mean(abs(v-v_sol))
    
    # Partial pivoting
    start = time()
    solutions[1] = PartialPivot(A,v)
    end = time()
    time_PP[i] = end - start
    v_sol = np.dot(A,solutions[1])
    err_PP[i] = np.mean(abs(v-v_sol))
    
    # LU decomposition
    start = time()
    solutions[2] = solve(A,v)
    end = time()
    time_LU[i] = end - start
    v_sol = np.dot(A,solutions[2])
    err_LU[i] = np.mean(abs(v-v_sol))

plt.figure(1)
plt.plot(N,np.log10(time_GE),label='Gauss elimination')
plt.plot(N,np.log10(time_PP),label='Partial pivoting')
plt.plot(N,np.log10(time_LU),label='LU decomposition')
plt.xlabel('size of the matrix')
plt.ylabel('magnitude of time')
plt.title('Plot of magnitude of solving time')
plt.legend(loc='lower right')


plt.figure(2)
plt.plot(N,np.log10(err_GE),label='Gauss elimination')
plt.plot(N,np.log10(err_PP),label='Partial pivoting')
plt.plot(N,np.log10(err_LU),label='LU decomposition')
plt.xlabel('size of the matrix')
plt.ylabel('magnitude of error')
plt.title('Plot of magnitude of error')
plt.legend(loc='upper left')

'''
# the data from Physics Background
R1=R3=R5=1e3
R2=R4=R6=2e3
C1=1e-6
C2=0.5e-6
xp=3
omega=1000

# write the linear system into matrices
A = np.array([[(1/R1+1/R4+1j*omega*C1),-1j*omega*C1,0],
               [-1j*omega*C1,(1/R2+1/R5+1j*omega*(C1+C2)),-1j*omega*C2],
               [0,-1j*omega*C2,(1/R3+1/R6+1j*omega*C2)]],complex)
v = np.array([xp/R1,xp/R2,xp/R3],complex)
# solve the linear system by partial pivoting
x = PartialPivot(A,v)

#calculate the amplitudes and phases of V1,V2,V3
Volt_amp = np.abs(x)
Volt_phase = np.angle(x)
print('The amplitudes of V1,V2,V3 are ',Volt_amp)
print('The phases of V1,V2,V3 are ',Volt_phase)

# plot the three voltages into functions of time
t = np.linspace(0,np.pi*2e-3,1000)
plt.figure(1)
plt.plot(t,np.real(x[0]*np.exp(1j*omega*t)),label='V1')
plt.plot(t,np.real(x[1]*np.exp(1j*omega*t)),label='V2')
plt.plot(t,np.real(x[2]*np.exp(1j*omega*t)),label='V3')
plt.legend()
plt.xlabel('time/sec')
plt.ylabel('voltage/V')
plt.title('Graph of V1,V2,V3')

# Replace R6 with an inductor L and repeat the steps
R6 *= 1j
A = np.array([[(1/R1+1/R4+1j*omega*C1),-1j*omega*C1,0],
               [-1j*omega*C1,(1/R2+1/R5+1j*omega*(C1+C2)),-1j*omega*C2],
               [0,-1j*omega*C2,(1/R3+1/R6+1j*omega*C2)]],complex)

# solve the linear system by partial pivoting
x = PartialPivot(A,v)

#calculate the amplitudes and phases of V1,V2,V3
Volt_amp = np.abs(x)
Volt_phase = np.angle(x)
print('\nAfter replace R6 with an inductor')
print('The amplitudes of V1,V2,V3 are ',Volt_amp)
print('The phases of V1,V2,V3 are ',Volt_phase)

# plot the three voltages into functions of time
plt.figure(2)
plt.plot(t,np.real(x[0]*np.exp(1j*omega*t)),label='V1')
plt.plot(t,np.real(x[1]*np.exp(1j*omega*t)),label='V2')
plt.plot(t,np.real(x[2]*np.exp(1j*omega*t)),label='V3')
plt.legend()
plt.xlabel('time/sec')
plt.ylabel('voltage/V')
plt.title('Graph of V1,V2,V3')
