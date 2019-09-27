# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 18:20:05 2018

@author: Heng
"""

from numpy import empty,zeros,max,exp
from pylab import imshow,gray,show
import numpy as np
import matplotlib.pyplot as plt
from dcst import dst,idst

# constants
L = 1.
d = 0.1
C = 1 
theta = 0.3
h = 1e-6
v = 100
num = 10000
a = L/num


# initial velocity psi0
def psi0(x):
    return C*(x*(L-x)/L**2)*exp(-(x-d)**2/(2*theta**2))

# the array of the piano string
pstr = np.arange(0,L+a,a)

#set the initial condition
psi = psi0(pstr)
phi = np.zeros(num+1)

phik = dst(phi)/num
psik = dst(psi)/num
omegak = pstr/a*np.pi*v/L

n = 0
N = 10000
plt.figure(figsize=(20,4))


t = 2e-3#Time spot for plots
phi_all = np.zeros([num+1,num+1],float)
for i in range(num):
    phi_all[i,:] = np.sin((i+1)*np.pi*pstr/L)*(phik[i+1]*np.cos(omegak[i+1]*t)+psik[i+1]/omegak[i+1]*np.sin(omegak[i+1]*t)) 
phi = phi_all.sum(axis = 0)
plt.ylim(-0.0005,0.0005)
plt.plot(pstr,phi)
plt.xlabel('x')
plt.ylabel('φ(x)')
plt.title('φ(x,t) for t = '+str(t*1e3)+' ms')
        
        
