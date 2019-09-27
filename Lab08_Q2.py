# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 10:52:49 2018

@author: Civet
"""

from numpy import empty,zeros,max,exp
from pylab import imshow,gray,show
import numpy as np
import matplotlib.pyplot as plt

L = 1
d = 0.1
C = 1 
theta = 0.3
h = 1e-6
v = 100
a = L/100

def velocity(x):
    return C*(x*(L-x)/L**2)*exp(-(x-d)**2/(2*theta**2))

while delta<N:
    plt.clf()
    boundary(T0)
    # Calculate new values of the potential
    for i in range(X+1):
            if i==0 or i==X :
                T0[i] = T0[i]
            else:
                T0[i] = (velocity(x)/a)**2
    boundary(T0)
    # Calculate maximum difference from old values
    delta +=1


    T0[51:150,0:30]=np.nan
    T = np.transpose(T0)
      # Make a plot
    plt.contourf(T,500)
    gray()
    plt.pause(0.01)