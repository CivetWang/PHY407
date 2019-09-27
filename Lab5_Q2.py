# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 17:41:43 2018

@author: student
"""

import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from numpy.fft import rfft2,irfft2

# read the image
fig=np.loadtxt("blur.txt")

# define the Gaussian function
def f(x,y):
    sigma = 25
    return exp(-(x**2+y**2)/(2*sigma**2))

#a. show the blurred image
plt.figure(1)
plt.imshow(fig,cmap='gray')
plt.title('Blurred image')

#b. create an array of periodic Guassian
rows = fig.shape[0]
cols = fig.shape[1]
gauss = np.zeros([rows,cols])
for m in range(rows): # use the loops from Computation background
    mt = m
    if mt > rows/2:
        mt -=rows
    for n in range(cols):
        nt = n
        if nt > cols/2:
            nt -=cols
        gauss[m,n] = f(mt,nt)

#Density plot of the array
plt.figure(2)
plt.imshow(gauss,cmap='gray')
plt.colorbar()
plt.title('Point spread function\n Gaussian with sigma=25')

#c.
figft = rfft2(fig) #Fourier transform of the image
gaussft = rfft2(gauss) #Fourier transform of the point spread function array
divft = np.ones([1024,513],complex) #create an array to be the quotient
e = 1e-3 
for i in range(1024): #to devide Fourier transform of the image by point spread function
    for j in range(513):
        if gaussft[i][j] < e: # to avoid divide by 0
            divft[i][j] = figft[i][j]/1024**2
        else:
            divft[i][j] = figft[i][j]/gaussft[i][j]/1024**2
#show the image after deconvolution
plt.figure(3)
plt.imshow(irfft2(divft),cmap='gray')
plt.title('Deconvoluted image')