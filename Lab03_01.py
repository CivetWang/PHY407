# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 17:08:38 2018

@author: student
"""

import numpy as np
from gaussxw import gaussxwab
from scipy.special import erf
import matplotlib.pyplot as plt

#Trapezoidal's rule
def Trap(func,N,a,b):
    h = (b-a)/N
    s = 0.5*func(a)+0.5*func(b)
    for k in range(1,N):
        s += func(a+k*h)
    return h*s
#Simpson's rule
def Simp(func,N,a,b):
    h = (b-a)/N
    s = 0
    for k in range(int(N/2)):
        s += 1./3 * (func(a+2*k*h)+4*func(a+(2*k+1)*h)+func(a+(2*k+2)*h))
    return h*s
#Gaussian Quadrature
def Gaus(func,N,a,b):
    xp,wp = gaussxwab(N,a,b)
    s = 0.0
    for k in range(int(N)):
        s += wp[k]*func(xp[k])
    return s
#the function to intergrate
def sample(t):
    return np.exp(-t**2)


print('the value using Trapezoidal\'s rule is ',2/np.sqrt(np.pi)*Trap(sample,8,0.,3.))
print('the value using Simpson\'s rule is ',2/np.sqrt(np.pi)*Simp(sample,8,0.,3.))
print('the value using Gaussian Quadrature is ',2/np.sqrt(np.pi)*Gaus(sample,8,0.,3.))

print('the value using Trapezoidal\'s rule is ',2/np.sqrt(np.pi)*Trap(sample,1000,0.,3.))
print('the value using Simpson\'s rule is ',2/np.sqrt(np.pi)*Simp(sample,1000,0.,3.))
print('the value using Gaussian Quadrature is ',2/np.sqrt(np.pi)*Gaus(sample,1000,0.,3.))


def rel_err(x,x0):
    return np.abs((x-x0))/x0
N = range(8,1001)
Trap_results = np.ones(len(N))
Simp_results = np.ones(len(N))
Gaus_results = np.ones(len(N))
for i in range(len(N)):
    Trap_results[i]=2/np.sqrt(np.pi)*Trap(sample,N[i],0.,3.)
    Simp_results[i]=2/np.sqrt(np.pi)*Simp(sample,N[i],0.,3.)
    Gaus_results[i]=2/np.sqrt(np.pi)*Gaus(sample,N[i],0.,3.)

plt.figure(1)
plt.plot(np.log10(N),np.log10(rel_err(Simp_results,erf(3))))
plt.plot(np.log10(N),np.log10(rel_err(Trap_results,erf(3))))
plt.plot(np.log10(N),np.log10(rel_err(Gaus_results,erf(3))))



#Question1(c) Blowing Snow
u10s = (6,8,10)
ths = (24,48,72)
Ta = np.linspace(-30,30,100)

def Prob(u10,Ta,th):
    u_bar = 11.2 + 0.365*Ta + 0.00706*Ta**2 + 0.9*np.log(th)
    delta = 4.3 + 0.145*Ta + 0.00196*Ta**2
    s = 1/np.pi * Gaus(sample,100,-u_bar/(np.sqrt(2)*delta),(u10-u_bar)/(np.sqrt(2)*delta))
    return 1/np.sqrt(np.pi) * s

from pylab import show,clf,plot,legend
clf()
colors = ('r','g','b')
lines = ('.','-',':')

plt.figure(2)

for (u10,color) in zip(u10s,colors):
    for (th,line) in zip(ths,lines):
        plot_str = color + line
        Prob_Gaus = np.ones(len(Ta))
        label0 = 'windspeed '+str(u10)+' and age '+str(th)+' hours'
        for i in range(len(Ta)):
            Prob_Gaus[i] = Prob(u10,Ta[i],th)
        plot(Ta,Prob_Gaus,plot_str,label=label0)
legend(prop={'size':9})
show()

for i in (8,146,297,435,586,732,889,932):
    print(2/np.sqrt(np.pi)*Trap(sample,i,0.,3.),'  ',2/np.sqrt(np.pi)*Simp(sample,i,0.,3.),'  ',2/np.sqrt(np.pi)*Gaus(sample,i,0.,3.))