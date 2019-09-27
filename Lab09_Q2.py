# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 16:21:22 2018

@author: Civet
"""

import numpy as np
import matplotlib.pyplot as plt
from banded import banded
import vpython as vp

L =1e-8
m = 9.109*1e-31 
ih = 1j*1.054571*1e-34
N = 1000
a = L/N
h=1e-18
a1 = 1 + h*ih/(2*m*a**2)
a2 = -h*ih/(4*m*a**2)
b1 = 1 - h*ih/(2*m*a**2)
b2 = h*ih/(4*m*a**2)
x0 = L/2
sigma = 1e-10
k = 5*1e10

def initial(x1):
    return np.exp(-(x1-x0)**2/(2*sigma**2))*np.exp(1j*k*x1)

psi = np.zeros(N+1,complex)
x = np.linspace(0,L,N+1)
psi[:] = initial(x)
psi[0] = psi[-1] = 0

A = np.empty([3,N],complex)
A[0,:]=a2
A[1,:]=a1
A[2,:]=a2
for i in range(N-1):
    v = b1*psi[1:N] + b2*(psi[2:N+1] + psi[0:N-1]) 
    psi[1:N] = banded(A,v,1,1)

plt.plot(np.real(psi))