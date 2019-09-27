# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 18:26:23 2018

@author: Civet
"""

from numpy import empty,zeros,max
from pylab import imshow,gray,show

# Constants
L = 20         # Grid squares on a side
W = 8
step = 0.1  
X = int(L/step)   
Y = int(W/step)    
target = 1e-6   # Target accuracy

# Create arrays to hold potential values
phi = zeros([Y+1,X+1],float)
phiprime = empty([Y+1,X+1],float)
# Main loop
delta = 1.0
while delta>target:
 
    # Calculate new values of the potential
    for i in range(Y+1):
        for j in range(X+1):
            if i==0 or i==Y or j==0 or j==X:
                phiprime[i,j] = phi[i,j]
            else:
                phiprime[i,j] = (phi[i+1,j] + phi[i-1,j] + phi[i,j+1] + phi[i,j-1])/4

    # Calculate maximum difference from old values
    delta = max(abs(phi-phiprime))

    # Swap the two arrays around
    phi,phiprime = phiprime,phi

# Make a plot
imshow(phi)
gray()
show()