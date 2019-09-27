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
#error estimation for Gaussian Quadrature
def err_est(func,N,a,b):
    return Gaus(func,2*N,a,b) - Gaus(func,N,a,b)

#for 1(a), print results with slices of a range of 8 to 1000
for i in (8,108,208,308,408,508,608,708,808,908):
    print(2/np.sqrt(np.pi)*Trap(sample,i,0.,3.),'  ',2/np.sqrt(np.pi)*Simp(sample,i,0.,3.),'  ',2/np.sqrt(np.pi)*Gaus(sample,i,0.,3.))

#relative error function
def rel_err(x,x0):
    return np.abs((x-x0))/x0
N = range(8,1001)
Trap_results = np.ones(len(N))
Simp_results = np.ones(len(N))
Gaus_results = np.ones(len(N))
err_estimate = np.ones(len(N))
err_calculation = np.ones(len(N))
for i in range(len(N)):
    Trap_results[i]=2/np.sqrt(np.pi)*Trap(sample,N[i],0.,3.)
    Simp_results[i]=2/np.sqrt(np.pi)*Simp(sample,N[i],0.,3.)
    Gaus_results[i]=2/np.sqrt(np.pi)*Gaus(sample,N[i],0.,3.)
    err_calculation[i]=rel_err(Gaus_results[i],erf(3))
    if err_calculation[i]==0:
        err_calculation[i] = 1e-16
    err_estimate[i]=err_est(sample,N[i],0.,3.)
    if err_estimate[i] == 0:
        err_estimate[i] = 1e-16

#plot magnitude of relative errors as functions of logN
plt.figure(1)
plt.plot(np.log10(N),np.log10(rel_err(Simp_results,erf(3))),label='Trapezoidal\'s rule')
plt.plot(np.log10(N),np.log10(rel_err(Trap_results,erf(3))),label='Simpson\'s rule')
plt.plot(np.log10(N),np.log10(err_calculation),label='Gaussian Quadrature')
plt.legend(loc=1)
plt.xlabel('log(N)')
plt.ylabel('magnitude of the relative error')
plt.title('Magnitude of relative errors as functions of logN')
plt.show()

#Compared with estimation
plt.figure(2)
plt.plot(np.log10(N),np.log10(rel_err(Gaus_results,erf(3))),label='calculated relative error')
plt.plot(np.log10(N),np.log10(err_estimate/erf(3)),label='estimated relative error')
plt.xlabel('log(N)')
plt.ylabel('magnitude of the relative error')
plt.legend()
plt.title('Comparison with error setimate')
plt.show()

#Question1(c) Blowing Snow
u10s = (6,8,10)
ths = (24,48,72)
Ta = np.linspace(-30,30,100)

#PRobability function
def Prob(u10,Ta,th):
    u_bar = 11.2 + 0.365*Ta + 0.00706*Ta**2 + 0.9*np.log(th)
    delta = 4.3 + 0.145*Ta + 0.00196*Ta**2
    s = 1/np.pi * Gaus(sample,100,-u_bar/(np.sqrt(2)*delta),(u10-u_bar)/(np.sqrt(2)*delta))
    return 1/np.sqrt(np.pi) * s

from pylab import show,clf,plot,legend,xlabel,ylabel,title
#plotting information
clf()
colors = ('r','g','b')
lines = ('.','-',':')

plt.figure(3)
#plot the probability
for (u10,color) in zip(u10s,colors):
    for (th,line) in zip(ths,lines):
        plot_str = color + line
        Prob_Gaus = np.ones(len(Ta))
        label0 = 'windspeed '+str(u10)+' and age '+str(th)+' hours'
        for i in range(len(Ta)):
            Prob_Gaus[i] = Prob(u10,Ta[i],th)
        plot(Ta,Prob_Gaus,plot_str,label=label0)
legend(prop={'size':9})
xlabel('average hourly temperature(Ta)/℃')
ylabel('Probability of blowing snow')
title('Graph of probability of blowing snow')
show()
  
