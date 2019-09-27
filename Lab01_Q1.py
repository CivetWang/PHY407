# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 16:27:59 2018

@author: hengl
"""

#1. First of first, import numpy and pyplot
import numpy as np
import matplotlib.pyplot as plt

#2. Set the time series t
dt = 0.0001
time = 1
t = np.linspace(0,time,time/dt)

#3. Set the constants
G = 39.5

#4. Set the variables as zero series with length of the time series
x = np.zeros(len(t))
y = np.zeros(len(t))
v_x = np.zeros(len(t))
v_y = np.zeros(len(t))

#5. Set the initial conditions
x[0] = 0.47
y[0] = 0
v_x[0] = 0
v_y[0] = 8.17

#6. Do the numerical integration by Euler-Cromer method
for i in range(len(t)-1):
    r = np.sqrt(x[i]**2+y[i]**2)
    v_x[i+1] = v_x[i]-G*x[i]/r**3*dt
    v_y[i+1] = v_y[i]-G*y[i]/r**3*dt
    x[i+1] = x[i] + v_x[i+1]*dt
    y[i+1] = y[i] + v_y[i+1]*dt

#7. Print the graph of components of velocity vs. time
plt.figure(1)
plt.plot(t,v_x)
plt.title('Graph of the x components of velocity')
plt.ylabel('v_x / (AU/yr)')
plt.xlabel('time / yr')

plt.figure(2)
plt.plot(t,v_y)    
plt.title('Graph of the y components of velocity')
plt.ylabel('v_y / (AU/yr)')
plt.xlabel('time / yr')
    
#8. Print the plot of the orbit in space    
plt.figure(3)
plt.plot(x,y)
plt.xlabel('x / AU')
plt.ylabel('y / AU')
plt.title('Orbit of the Mercurry using Newtonian gravitational force')

#9. Check if angular momentum is conserved
plt.figure(4)
AM = v_y*x-v_x*y
plt.plot(t,AM)
plt.title('Graph of the angular momentum')
plt.ylim(3,5)
plt.ylabel('angular momentum / (Ms*AU^2/yr)')
plt.xlabel('time / yr')

#10. Repeat the steps again but using general-relativity gravitational force
dt = 0.001
time = 10
t = np.linspace(0,time,time/dt)
x = np.zeros(len(t))
y = np.zeros(len(t))
v_x = np.zeros(len(t))
v_y = np.zeros(len(t))

x[0] = 0.47
y[0] = 0
v_x[0] = 0
v_y[0] = 8.17

alpha = 0.01 # use alpha = 0.01 AU^2 to show the effect more obviously

for i in range(len(t)-1):
    r = np.sqrt(x[i]**2+y[i]**2)
    v_x[i+1] = v_x[i]-G*x[i]*(1+alpha/r**2)/r**3*dt
    v_y[i+1] = v_y[i]-G*y[i]*(1+alpha/r**2)/r**3*dt
    x[i+1] = x[i] + v_x[i+1]*dt
    y[i+1] = y[i] + v_y[i+1]*dt
    
plt.figure(5)
plt.plot(x,y)
plt.xlabel('x / AU')
plt.ylabel('y / AU')
plt.title('Orbit of the Mercurry using general-relativity-form gravitational force in 10 years')