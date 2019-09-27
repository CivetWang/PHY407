# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 20:41:58 2018

@author: student
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from numpy import empty,zeros,max
from pylab import imshow,gray,show
import numpy as np
import matplotlib.pyplot as plt

# Constants
L = 20
W = 8
step = 0.1
X = int(L/step)
Y = int(W/step)
V = 1.0         # Voltage at top wall
target = 1e-6   # Target accuracy

def boundary(A):
    A[51:150,0:30]=0
    A[0:51,0] = np.linspace(0.,5.,len(A[0:51,0]))
    A[150:,0] = np.linspace(5.,0.,len(A[150:,0]))
    A[50,0:31] = np.linspace(5.,7.,len(A[50,0:31]))
    A[150,0:31] = np.linspace(5.,7.,len(A[50,0:31]))
    A[0,:] = np.linspace(0.,10,len(A[0,:]))
    A[-1,:] = np.linspace(0.,10,len(A[-1,:]))
    A[50:151,30] = 7
    A[:,-1] = 10
    return A

     
# Create arrays to hold potential values
T0 = zeros([X+1,Y+1],float)
Tp = zeros([X+1,Y+1],float)

# Main loop
delta = 1.0
N=100
while delta<N:
    plt.clf()
    boundary(T0)
    # Calculate new values of the potential
    for i in range(X+1):
        for j in range(Y+1):
            if i==0 or i==X or j==0 or j==Y:
                Tp[i,j] = T0[i,j]
            else:
                Tp[i,j] = (T0[i+1,j] + T0[i-1,j] + T0[i,j+1] + T0[i,j-1])/4
    boundary(T0)
    # Calculate maximum difference from old values
    Tp,T0 = T0,Tp


    T0[51:150,0:30]=np.nan
    T = np.transpose(T0)
      # Make a plot
    plt.contour(T,500)
    gray()
    plt.pause(0.01)