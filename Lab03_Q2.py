# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 16:53:41 2018

@author: student
"""
# import all necessary modules
import numpy as np
from math import factorial
from pylab import plot, show, clf 
from gaussxw import gaussxwab
import matplotlib.pyplot as plt
#Q1 Harmonic oscillator
#By definition to set Hermite polynomial function
def H(n,x):
    H_x=np.zeros(n+1)
    H_x[0] = 1
    if n>=1:
       H_x[1] = 2*x
    i=2
    while i <=n:
        H_x[i] = 2*x*H_x[i-1]-2*n*H_x[i-2]
        i +=1
    return H_x[n]
#By defination set Wavefunction for Harmonic oscillator
def psi_n(n,x):
    a = np.sqrt(float(2**n)*float(factorial(n))*np.sqrt(np.pi))
    b= np.exp(-x**2/2)
    return (1/a)*b*H(n,x)

#clear any potential image in console
clf()
x = np.linspace(-4,4,120)#space for all x around all four state
lines = ('.', '-', ':','-.')
colors = ['r','g','b','y']
ns = [0,1,2,3]
for i in range(len(ns)):
    y=np.zeros(len(x))
    label0 = 'n='+str(ns[i])+' with '+str(colors[i])+' color'
    for j in range(len(x)):
        y[j]=psi_n(ns[i],x[j])
    plot(x,y,colors[i],label=label0)
plt.title('n = 0,1,2,3 wave function')
plt.ylabel('Wavefunction amplitude')
plt.xlabel('x')
plt.legend(loc=0,fontsize='small')
show()
#Q2 state=30
clf()
x = np.linspace(-10,10,2000)#state of n=30 around the domain
y=np.zeros(len(x))
for i in range(len(x)):
        y[i] = psi_n(30,x[i])
plot(x,y,label='n=30')
plt.title('n = 30 wave function')
plt.ylabel('Wavefunction amplitude')
plt.xlabel('x')
plt.legend(loc=0,fontsize='large')
show()

#function upon change of variable
def H_t(n,x):
    H_x=np.zeros(n+1)
    H_x[0] = 1
    if n>=1:
       H_x[1] = 2*np.tan(x)
    i=2
    while i <=n:
        H_x[i] = 2*np.tan(x)*H_x[i-1]-2*(i-1)*H_x[i-2]
        i +=1
    return H_x[n]
#function upon change of variable
def psi_n_t(n,x):
    a = np.sqrt(float(2**n)*float(factorial(n))*np.sqrt(np.pi))
    b= np.exp(-np.tan(x)**2/2)
    return (1/a)*b*H_t(n,x)
#function upon change of variable
def dpsi_n(n,x):
    a = np.sqrt(float(2**n)*float(factorial(n))*np.sqrt(np.pi))
    b= np.exp(-np.tan(x)**2/2)
    if n == 0:
        c= -np.tan(x)
    else:
        c= -np.tan(x)*H_t(n,x)+2*n*H_t(n-1,x)
    return (1/a)*b*c
#define the uncertainty of position
def unce_x(n,x):
    return (np.tan(x)**2*abs(psi_n_t(n,x))**2)/np.cos(x)**2
#define the uncertainty of momentum
def unce_p(n,x):
    return (abs(dpsi_n(n,x)))**2/np.cos(x)**2
#define energy
def En(n,x):
    return (unce_p(n,x)+unce_x(n,x))*0.5

def Gaus(func,N,n,a,b):
    xp,wp = gaussxwab(N,a,b)
    s = 0.0
    for k in range(int(N)):
        s += wp[k]*func(n,xp[k])
    return s

#draw the uncertainty diagram
n= np.linspace(0,15,16)
x = np.zeros(16)
p = np.zeros(16)
E = np.zeros(16)
for i in range(len(x)):
    x[i] = np.sqrt(Gaus(unce_x,100,i,-np.pi/2,np.pi/2))
    p[i] = Gaus(unce_p,100,i,-np.pi/2,np.pi/2)
    print('uncertainty of position is '+str(x[i])+',the uncertainty in momentum is '+str(p[i]))
for i in range(len(E)):
    E[i]= (x[i]**2+p[i])*0.5
    print('The energy is '+str(E[i]), 'for n = ',i)

plot(x,p,label='P/x')
plt.title('uncertainty comparison')
plt.ylabel('uncertainty<P^2>')
plt.xlabel('uncertainty√<x^2>')
plt.legend(loc=0,fontsize='large')
plt.figure(2)
plot(n,E,label='E/n')
plt.title('the energy of the oscillator')
plt.ylabel('the energy of the oscillator>')
plt.xlabel('state n')
plt.legend(loc=0,fontsize='large')