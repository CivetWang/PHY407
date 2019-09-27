# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 16:26:48 2018

@author: Heng
"""

import numpy as np
import matplotlib.pyplot as plt

E = 1
dx = 0.02
dt = 0.005
Lx = 2*np.pi
Tf = 2
N = 1000
beta = E*dt/dx

u = np.zeros([N,int(Lx/dx)+2])
x = np.arange(0,Lx+dx,dx)
u[0,:] = np.sin(x)
u[0,0] = 0
u[0,-1] = 0

u[1,1:-1] = u[0,1:-1] - beta/2*(u[0,2:]**2-u[0,:-2]**2)
u[1,0] = 0
u[1,-1] = 0



for i in range(N-2):
    j = i+2
    u[j,1:-1] = u[j-2,1:-1] - beta/2*(u[j-1,2:]**2-u[j-1,:-2]**2)
    u[j,0] = 0
    u[j,-1] = 0
    plt.clf()
    plt.plot(x,u[j])
    plt.pause(0.01)


    