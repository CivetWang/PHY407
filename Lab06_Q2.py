# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 17:29:27 2018

@author: student
"""

#import moduleds
import numpy as np 
import matplotlib.pyplot as plt

#Set constants
sigma = 1
epsilon = 1

#set the initial conditions
r1 = np.array([4.,4.])
r2 = np.array([5.2,4.])
v = np.array([0.,0.,0.,0.])

#Set timing step
t1 = 0
dt = 0.01
N = 100
t2 = t1 + N*dt
r= np.append(r1,r2)

#prepare the timeline array and data array
tpoints = np.arange(t1,t2,dt)
x1points=[]
x2points=[]
y1points=[]
y2points=[]

vx1points=[]
vx2points=[]
vy1points=[]
vy2points=[]

#define potential function and other relative funcs
def Vp(r):
    return 2*epsilon*(-12*sigma**12/r**13+6*sigma**6/r**7)
def V(r):
    return 4*epsilon*((sigma/r)**12-(sigma/r)**6)

def f(r,t):
    x1=r[0]
    y1=r[1]
    x2=r[2]
    y2=r[3]
    r0 = np.sqrt((x1-x2)**2+(y1-y2)**2)
    f1=-Vp(r0)*(x1-x2)/r0/2
    f2=-Vp(r0)*(y1-y2)/r0/2
    f3=-Vp(r0)*(x2-x1)/r0/2
    f4=-Vp(r0)*(y2-y1)/r0/2
    return np.array([f1,f2,f3,f4],float)

vhalf = v + 0.5*dt*f(r,0)
for t in tpoints:
    x1points.append(r[0])
    y1points.append(r[1])
    x2points.append(r[2])
    y2points.append(r[3])
    vx1points.append(v[0])
    vy1points.append(v[1])
    vx2points.append(v[2])
    vy2points.append(v[3])
    r += vhalf*dt
    k = dt*f(r,t+dt)
    v = vhalf + 0.5*k
    vhalf += k

#plot output
plt.figure(1)
plt.plot(x1points,y1points,'r.',markersize=1)
plt.plot(x2points,y2points,'b.',markersize=1)
plt.figure(2)
plt.plot(tpoints,x1points)
plt.plot(tpoints,x2points)
plt.figure(3)
E = V(np.sqrt((np.array(x1points)-np.array(x2points))**2+(np.array(y1points)
    -np.array(y2points))**2))+0.5*(np.array(vx1points)**2+np.array(vy1points)**2
    +np.array(vx2points)**2+np.array(vy2points)**2)
plt.plot(tpoints,E)