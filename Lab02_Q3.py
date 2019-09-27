# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 09:40:33 2018

@author: Civet
"""
#1.First of first, import numpy and constants function for final check
import numpy as np
from scipy import constants
#2. Set up all given constants for equation usage
rPc=1.054571800e-34 # redused Planck Constant
kB = 1.38064852e-23# boltzmann constant
c = 299792458# speed of light
#Trapezoidal's rule
def Trap(func,N,a,b):
    h = (b-a)/N#make slicing
    temp = np.linspace(a+h,b-h,N-1)#correct the array along the axis
    s = 0.5*func(a)+0.5*func(b)
    s += sum(func(temp))
    return h*s

#Simpson's rule
def Simp(func,N,a,b):
    h = (b-a)/N#make slicing
    s = 0
    for k in range(int(N/2)):
        s += 1./3 * (func(a+2*k*h)+4*func(a+(2*k+1)*h)+func(a+(2*k+2)*h))
        #correct the array along the axis
    return h*s
# integral function
def sample(x):
    return (x**3)/(np.exp(x)-1)
def sample_1(x):
    return (np.tan(x)**3)/((np.exp(np.tan(x))-1)*np.cos(x)**2)
#Output two answer for two methods.
#Trapezoidal's rule
print('the value using Trapezoidal\'s rule is ',Trap(sample_1,100,0.01,np.pi/2-0.01))
#practical estimation of errors for Trapezoidal's rule
error_Trap = 4./3 *(Trap(sample_1,100*2,0.01,np.pi/2-0.01) - Trap(sample_1,100,0.01,np.pi/2-0.01))
print('the practical estimation of error for Trapezoidal\'s rule when N=100 is ',error_Trap)

#Simpson's rule
print('the value using Simpson\'s rule is ',Simp(sample_1,100,0.01,np.pi/2-0.01))
#practical estimation of errors for Simpson's rule
error_Simp = 4./15 * (Simp(sample_1,100*2,0.01,np.pi/2-0.01) - Simp(sample_1,100,0.01,np.pi/2-0.01))
print('the practical estimation of error for Simpson\'s rule when N=10 is ',error_Simp)  

#Evaluate with all settled number and compute the constant to compare with the value in scipy
constant = Trap(sample_1,1000,0.01,np.pi/2-0.01)*kB**4/(4*np.pi**2*c**2*rPc**3)
print('the result we compute is',constant)
results= constants.physical_constants['Stefan-Boltzmann constant']
print('the value from scipy is',results[0],'+-', results[2])