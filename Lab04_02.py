# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 01:00:41 2018

@author: Heng
"""

from numpy import array
from numpy.linalg import eigh
import numpy as np
from scipy.constants import hbar
import matplotlib.pyplot as plt


# write the Hmatrix function given in the Physics Background
def Hmatrix(m,n):
    a = 10 * 1.6022e-19
    L = 5e-10
    M = 9.1094e-31
    h = 0
    if m==n:
        h = 0.5*a+np.pi**2*hbar**2*m**2/(2*M*L**2)
    if (m!=n) & ((m%2)!=(n%2)):
        h = -8*a*m*n/(np.pi**2*(m**2-n**2)**2)
    return h
#question(c)       
mmax=10
nmax=10
H = np.ones([mmax,nmax])
a = 10 * 1.6022e-19
L = 5e-10

for m in range(1,mmax+1):
    for n in range(1,nmax+1):
        H[m-1,n-1] = Hmatrix(m,n)

En,V = eigh(H)
En /= 1.6022e-19
print(En)

#question(d)
#set the size of matrix
mmax = nmax = 100
H = np.ones([mmax,nmax])
# calculate H
for m in range(1,mmax+1):
    for n in range(1,nmax+1):
        H[m-1,n-1] = Hmatrix(m,n)
#calculate En and its eigenvectors
En,V = eigh(H)
En /= 1.6022e-19
print(En[:9])


#define psi_square for n
def psi_square(x,n):
    s = 0
    for i in range(1,mmax+1):
        s += V[n][i-1]*np.sin(np.pi*i/L*x)
    return s**2

def psi_square0(x):
    return psi_square(x,0)

def psi_square1(x):
    return psi_square(x,1)

def psi_square2(x):
    return psi_square(x,2)
#Simpson's rule
def Simp(func,N,a,b):
    h = (b-a)/N
    s = 0
    for k in range(int(N/2)):
        s += 1./3 * (func(a+2*k*h)+4*func(a+(2*k+1)*h)+func(a+(2*k+2)*h))
    return h*s
#plot the probability density
x = np.linspace(0,L,1000)
plt.plot(x,psi_square(x,0)/Simp(psi_square0,1000,0,L),label='Ground State')
plt.plot(x,psi_square(x,1)/Simp(psi_square1,1000,0,L),label='1st excited state')
plt.plot(x,psi_square(x,2)/Simp(psi_square2,1000,0,L),label='2nd excited state')
plt.xlabel('x/m')
plt.ylabel('Probability')
plt.title('Graph of probability density')
plt.legend(loc='upper right',prop={'size':7})