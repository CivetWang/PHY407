# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 16:22:31 2018

@author: Civet
"""

"""
Use the program to simulate the circulation motion of object
about the ball-rod system which describe the space garbage system
"""
#import moduleds
import numpy as np 
import pylab as plt
import time 
t0 = time.clock()
#Set G, M, L constants
G=1
M=10
L=2

#define the ODE system
def f(r,t):
    xp=r[0]
    x=r[1]
    yp=r[2]
    y=r[3]
    r0 = np.sqrt(x**2+y**2)
    f1=-G*M*x/(r0**2*np.sqrt(r0**2+L**2/4))
    f2=xp
    f3=-G*M*y/(r0**2*np.sqrt(r0**2+L**2/4))
    f4=yp
    return np.array([f1,f2,f3,f4],float)

#Set timing step
t1 = 0
t2 = 10
N = 10000
h = (t2-t1)/N

#prepare the timeline array
tpoints = np.arange(t1,t2,h)
xppoints=[]
xpoints=[]
yppoints=[]
ypoints=[]

#prepare the initial condition of the system 
r= np.array ([0.0,1.0,1.0,0.0],float)

#Usw Runge_Kutta method to solve the system
for t in tpoints:
    xppoints.append(r[0])
    xpoints.append(r[1])
    yppoints.append(r[2])
    ypoints.append(r[3])
    k1 = h*f(r,t)
    k2 = h*f(r+0.5*k1,t+0.5*h)
    k3 = h*f(r+0.5*k2,t+0.5*h)
    k4 = h*f(r+k3,t+h)
    r += (k1+2*k2+2*k3+k4)/6
#output
plt.plot(xpoints,ypoints)
plt.xlabel('Position in X')
plt.ylabel('Position in Y')
plt.title('Space Garbage system overtime 10')
plt.show()
print (time.clock()-t0)
np.savetxt('Lab6Q1x.txt',np.column_stack(np.array(xpoints)))
np.savetxt('Lab6Q1y.txt',np.column_stack(np.array(ypoints)))