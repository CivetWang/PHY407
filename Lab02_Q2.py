# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 16:13:29 2018

@author: hengl
"""

import scipy as sp
import numpy as np
from scipy.special import erf
from time import time
import matplotlib.pyplot as plt

#Trapezoidal's rule
def Trap(func,N,a,b):
    h = (b-a)/N
    temp = np.linspace(a+h,b-h,N-1)
    s = 0.5*func(a)+0.5*func(b)
    s += sum(func(temp))
    return h*s

#Simpson's rule
def Simp(func,N,a,b):
    h = (b-a)/N
    s = 0
    for k in range(int(N/2)):
        s += 1./3 * (func(a+2*k*h)+4*func(a+(2*k+1)*h)+func(a+(2*k+2)*h))
    return h*s
#Euler-Maclaurin formula for Trapezoidal's rule
def EM_Trap(func_firstorder,N,a,b):
    h = (b-a)/N
    return 1./12*h**2*(func_firstorder(a)-func_firstorder(b))
#Euler-Maclaurin formula for Simpson's rule
def EM_Simp(func_thirdorder,N,a,b):
    h = (b-a)/N
    return 1./180*h**4*(func_thirdorder(a)-func_thirdorder(b))
    
'''
Question A: evaluate the erf function

'''    
def sample(t):
    return np.exp(-t**2)

def sample_firstorder(t):
    return -2*np.exp(-t**2)*t

def sample_thirdorder(t):
    return -4*np.exp(-t**2)*t*(2*t**2-3)

print('the value using Trapezoidal\'s rule is ',2/np.sqrt(np.pi)*Trap(sample,10,0,3))
print('the value using Simpson\'s rule is ',2/np.sqrt(np.pi)*Simp(sample,10,0,3))
print('the value using erf function is ',erf(3))

count = 1
while (np.abs(2/np.sqrt(np.pi)*Trap(sample,10**count,0,3)-erf(3))/erf(3)>10**-11):
    count += 1

print('need ',count,' slices for Trapezoidal\'s rule')

count_Trap = count

count = 1
while (np.abs(2/np.sqrt(np.pi)*Simp(sample,10**count,0,3)-erf(3))/erf(3)>10**-11):
    count += 1

print('need ',count,' slices for Simpson\'s rule')

count_Simp = count
#calculate how long it takes to compute the intergral for Traapezoidal's rule 
start = time()
2/np.sqrt(np.pi)*Trap(sample,10**count_Trap,0,3)
end = time()
time_Trap = end - start

#calculate how long it takes to compute the intergral for Simpson's rule 
start = time()
2/np.sqrt(np.pi)*Trap(sample,10**count_Simp,0,3)
end = time()
time_Simp = end - start

#calculate how long it takes to compute the intergral for scipy's function
start = time()
erf(3)
end = time()
time_Scipy = end - start

print(time_Trap,'\n',time_Simp,'\n',time_Scipy)

#practical estimation of errors for Trapezoidal's rule
error_Trap = 4./3 * (Trap(sample,10*2,0,3) - Trap(sample,10,0,3))
print('the practical estimation of error for Trapezoidal\'s rule when N=10 is ',error_Trap)
#practical estimation of errors for Simpson's rule
error_Simp = 4./15 * (Simp(sample,10*2,0,3) - Simp(sample,10,0,3))
print('the practical estimation of error for Simpson\'s rule when N=10 is ',error_Simp)    

#results of Euler-Maclaurin formulas for both method
error_EM_Trap = EM_Trap(sample_firstorder,10,0,3)
error_EM_Simp = EM_Simp(sample_firstorder,10,0,3)
print('result of Euler-Maclaurin formula for Trapezoidal\'s rule when N=10 is ',error_EM_Trap)
print('result of Euler-Maclaurin formula for Simpson\'s rule when N=10 is ',error_EM_Simp)

'''
Question B: Diffraction limit of a telescope
'''
from scipy import special

#Question 2(b)
#Excercise 5.4a
#Write the J(m,x) function using Simpson's rule
def J(m,x):
    N = 1000
    def temp(theta):
        return np.cos(m*theta-x*np.sin(theta))
    return 1./np.pi * Simp(temp,N,0,np.pi)
x = np.linspace(0,20,1000)
plt.figure(1)
plt.plot(x,J(0,x),label='J0')
plt.plot(x,J(1,x),label='J1')
plt.plot(x,J(2,x),label='J2')
plt.title('Graph of Bessel functions')
plt.ylabel('Value of Bessel functions')
plt.xlabel('x')
plt.legend()

#plot the differences between reproduced Bessel functions and functions in scipy
plt.figure(2)
plt.plot(x,special.jv(0,x)-J(0,x),label='J0')
plt.title('Difference of Bessel function J0')
plt.ylabel('Difference')
plt.xlabel('x')
plt.figure(3)
plt.plot(x,special.jv(1,x)-J(1,x),label='J1')
plt.title('Difference of Bessel function J1')
plt.ylabel('Difference')
plt.xlabel('x')
plt.figure(4)
plt.plot(x,special.jv(2,x)-J(2,x),label='J2')
plt.title('Difference of Bessel function J2')
plt.ylabel('Difference')
plt.xlabel('x')

#Excercise 5.4b
#write light intensity into function
def light_intensity(r):
    k = 2*np.pi/(500e-9)
    return (J(1,k*r)/k*r)**2
size = 10 #choose the size of the array
X = np.linspace(-1e-6,1e-6,size)
Y = np.linspace(-1e-6,1e-6,size)
intensity = np.ones((size,size))
#calculate the light intensity
for i in range(size):
    for j in range(size):
        intensity[i,j] = light_intensity(np.sqrt(X[i]**2+Y[j]**2))

#plot the density plot
plt.figure(5)
plt.pcolormesh(X*1e6,Y*1e6,intensity)
my_x_ticks = np.arange(-1, 1, 0.2)
my_y_ticks = np.arange(-1, 1, 0.2)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.title('Graph of the circular diffraction pattern of a point light sourse with λ=500nm')
plt.ylabel('y/μm')
plt.xlabel('x/μm')
plt.savefig("Graph of circular diffraction.png",dpi=300)