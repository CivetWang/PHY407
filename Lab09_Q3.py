# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 16:23:12 2018

@author: Civet
"""

import numpy as np
import matplotlib.pyplot as plt

# constants
E = 1
dx = 0.02
dt = 0.01
Lx = 2*np.pi
Tf = 2
N = 150
beta = E*dt/dx

#set the initial condition
u = np.zeros([N,int(Lx/dx)+2])
x = np.arange(0,Lx+dx,dx)
u[0,:] = np.sin(x)
u[0,0] = 0
u[0,-1] = 0

#main loop
for t in range(N-1):
    j = t+1
    uji=u[j-1,1:-1]#u(x,t)
    ujip=u[j-1,2:]#u(x,t+1)
    ujim=u[j-1,:-2]#u(x,t-1)
    a = uji-beta/4*(ujip**2-ujim**2)
    b = beta**2/8*((ujip+uji)**2*(ujip-uji)+(ujim+uji)**2*(ujim-uji))
    u[j,1:-1] = a + b#main calculation
    u[j,0] = 0#reset the boundary 
    u[j,-1] = 0#reset the boundary 
    plt.clf()
    plt.plot(x,u[j])
    plt.pause(0.01)
plt.figure(2)
plt.plot(x,u[0],label = '0s')
plt.plot(x,u[49],label = '0.5s')
plt.plot(x,u[99],label = '1s')
plt.plot(x,u[149],label = '1.5s')
plt.title('All four phases')
plt.legend()