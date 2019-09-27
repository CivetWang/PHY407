# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 11:40:05 2018

@author: Heng
"""

import struct
import numpy as np
import matplotlib.pyplot as plt

#1. open the data file
f = open('N19W156.hgt','rb')
#2 set the arrays 
W = np.ones((1201,1201))
Grad_x = np.ones((1201,1201))
Grad_y = np.ones((1201,1201))
X = np.linspace(0,1200,1201)
Y = np.linspace(0,1200,1201)

#3 write the data into the data array by loops, and check the bad points
for i in range(1201):
    for j in range(1201):
        buf = f.read(2)
        W[-i][j] = struct.unpack('>h',buf)[0]
        if W[-i][j]<-10:
            W[-i][j] = 0

h = 420
phi = np.pi

#4 calculate the partial derivative along y axis    
#use forward and backward for borders and central differences for other points
for j in range(1201):
    Grad_y[0][j] = (W[1][j]-W[0][j])/h
    Grad_y[-1][j] = (W[-1][j]-W[-2][j])/h
    for i in range(1,1200):
        Grad_y[i][j] = (W[i+1][j] - W[i-1][j])/(2*h)

#5 calculate the partial derivative along x axis    
#use forward and backward for borders and central differences for other points        
for i in range(1201):
    Grad_x[i][0] = (W[i][1]-W[i][0])/h
    Grad_x[i][-1] = (W[i][-1]-W[i][-2])/h
    for j in range(1,1200):
        Grad_x[i][j] = (W[i][j+1] - W[i][j-1])/(2*h)

#6 calculate I for every point
I = -(np.cos(phi)*Grad_x + np.sin(phi)*Grad_y)/np.sqrt((Grad_x)**2+(Grad_y)**2+1)

'''
for i in range(1201):
    for j in range(1201):
        if (I[i][j]<0) or (I[i][j]>0.1):
            I[i][j]=0
'''
#7 plot W and I by pcolormesh
plt.figure(1)
plt.pcolormesh(X,Y,W)
plt.colorbar()
plt.title('Plot of the elevation above sea level in metres')
plt.xlabel('x')
plt.ylabel('y')

plt.figure(2)
plt.pcolormesh(X,Y,I,vmax=0.025,vmin=0.,shading='gouraud')
plt.colorbar()
plt.title('Plot of the intensity of illumination of the surface at sunrise')
plt.xlabel('x')
plt.ylabel('y')
