# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 10:23:52 2018

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
delta=1e-6
#define the ODE system
def f(r):
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

def rho(r1,r2,h):
    return h*delta/np.sqrt((1/30.*(r1[1]-r2[1]))**2+(1/30.*(r1[3]-r2[3]))**2)

def Runge_Kutta(ri,h0):
    rp=ri
    k1 = h0*f(rp)
    k2 = h0*f(rp+0.5*k1)
    k3 = h0*f(rp+0.5*k2)
    k4 = h0*f(rp+k3)
    rp = rp+(k1+2*k2+2*k3+k4)/6.
    return rp

#Set timing step
t1 = 0
t2 = 10
h = 0.01

#prepare the timeline array
tpoints=[]
tpoints.append(t1)
xppoints=[]
xpoints=[]
yppoints=[]
ypoints=[]

#prepare the initial condition of the system 
r= np.array ([0.0,1.0,1.0,0.0],float)
xppoints.append(r[0])
xpoints.append(r[1])
yppoints.append(r[2])
ypoints.append(r[3])

#Usw Runge_Kutta method to solve the system
i=0
while tpoints[i] <= t2:
    r1 = Runge_Kutta(r,h)
    r1 = Runge_Kutta(r1,h)
    r2 = Runge_Kutta(r,2*h)
    if rho(r1,r2,h) <1.0:
        hp = h*rho(r1,r2,h)**(1/4)
    else:
        hp=2*h
    r = Runge_Kutta(r,hp)
    tpoints.append(tpoints[i]+hp)
    i +=1
    h = h*rho(r1,r2,h)**(1/4)
    xppoints.append(r[0])
    xpoints.append(r[1])
    yppoints.append(r[2])
    ypoints.append(r[3])
    
#output the diagram
plt.figure(1)
plt.plot(xpoints,ypoints,'k.',label='Adapted')
x=np.loadtxt('Lab6Q1x.txt')
y=np.loadtxt('Lab6Q1y.txt')
plt.plot(x,y,label='non-adapted')
plt.xlabel('Position in X')
plt.ylabel('Position in Y')
plt.title('Space Garbage system by adapted stepsize')
plt.legend()
plt.show()
print (time.clock()-t0)
plt.figure(2)
dtpoints = np.array(tpoints[1:])-np.array(tpoints[:-1])
plt.plot(tpoints[:-1],dtpoints)
plt.xlabel('Time')
plt.ylabel('Adapted stepsize')
plt.title('Adapted stepsize as a function of time')
plt.show()



