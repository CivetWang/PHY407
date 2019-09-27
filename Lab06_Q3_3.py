# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 11:55:25 2018

@author: hengl
"""
#import moduleds
import numpy as np
import matplotlib.pyplot as plt

#Set constants
sigma = 1
epsilon = 1
N=16
Lx=4
Ly=4

#set the initial conditions
dx=Lx/np.sqrt(N)
dy=Ly/np.sqrt(N)
x_grid=np.arange(dx/2,Lx,dx)
y_grid=np.arange(dy/2,Ly,dy)
xx_grid,yy_grid = np.meshgrid(x_grid,y_grid)
x_ini=xx_grid.flatten()
y_ini=yy_grid.flatten()
vx_ini = np.zeros(N)
vy_ini = np.zeros(N)

#Set timing step
t1 = 0
dt = 0.01
n = 1000
t2 = t1 + n*dt
tpoints = np.arange(t1,t2,dt)

#define periodic function 
def Periodic(r):
    r[0,:] = np.mod(r[0,:],Lx*np.ones(N))
    r[1,:] = np.mod(r[1,:],Ly*np.ones(N))
    return r

#define potential function and other relative funcs
def Vp(r):
    return 4*epsilon*(-12*sigma**12/r**13+6*sigma**6/r**7)
def V(r):
    return 4*epsilon*((sigma/r)**12-(sigma/r)**6)

Period = 4*np.array([[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]],float)

#define the processing method
def f(r,t):
    x = r[0,:]
    y = r[1,:]
    r0 = np.zeros([N,N])
    fx = np.zeros(N)
    fy = np.zeros(N)
    for i in range(N):
        for j in [m for m in range(N) if m!= i]:
            r0[i][j] = np.sqrt((x[i]-x[j])**2+(y[i]-y[j])**2)
            fx[i] += -Vp(r0[i][j])*(x[i]-x[j])/r0[i][j]
            fy[i] += -Vp(r0[i][j])*(y[i]-y[j])/r0[i][j]            
        for a in range(8):
            xp = r[0,:] + Period[a][0]
            yp = r[1,:] + Period[a][1]
            rp = np.zeros(N)
            for j in range(N):
                rp[j] = np.sqrt((x[i]-xp[j])**2+(y[i]-yp[j])**2)
                fx[i] += -Vp(rp[j])*(x[i]-xp[j])/rp[j]
                fy[i] += -Vp(rp[j])*(y[i]-yp[j])/rp[j]
    return np.array([fx,fy],float)

r = np.array([x_ini,y_ini])
v = np.array([vx_ini,vy_ini])


#define the processing method
vhalf = v + 0.5*dt*f(r,0)
xt = np.zeros([N,n])
yt = np.zeros([N,n])
vxt = np.zeros([N,n])
vyt = np.zeros([N,n])
Ek = np.zeros(n)
Ep = np.zeros(n)

#process through all 16 particles
for i in range(n):
    t = tpoints[i]
    xt[:,i] = r[0,:]
    yt[:,i] = r[1,:]
    vxt[:,i] = v[0,:]
    vyt[:,i] = v[1,:]
    for p in range(N):
        Ek[i] += 0.5*(vxt[p][i]**2+vyt[p][i]**2)  
        for q in [m for m in range(N) if m!= p]:
            Ep[i] += 0.5*V(np.sqrt((xt[p][i]-xt[q][i])**2+(yt[p][i]-yt[q][i])**2))
  
    r += vhalf*dt
    Periodic(r)
    k = dt*f(r,t+dt)
    v = vhalf + 0.5*k
    vhalf += k
    
#output the plot
plt.figure(1)
for i in range(N):
    plt.plot(xt[i,:],yt[i,:],':',markersize=1)
plt.xlabel('Position in X')
plt.ylabel('Position in Y')
plt.title('16 particles in a 4x4 frame with the boundary condition')
plt.show()

