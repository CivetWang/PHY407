# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 14:08:43 2018

@author: student
"""
#packages for use
import numpy as np
import numpy.random as ra
import matplotlib.pyplot as plt


a = 0#lower bound of integral
b = 10#Higher bound of integral
N = 10000#random total goal

# define the target function
def fx(x):
    return np.exp(-2*abs(x-5))

#define the weight function
def w(x):
    return 1/np.sqrt(2*np.pi)*np.exp(-(x-5)**2/2)

#define the mean value method
def generate (N):
    total = 0
    for i in range(N):
        x = ra.random(1)*10
        total += fx(x)
    return (b-a)*total/N

#prepare the set for containing result
result = np.zeros(100)

#combine the results into one array
for i in range(100):
    result[i]=generate(N)
    
#output the figure of the mean value method
plt.figure(1)  
plt.hist(result,10,range=[0.96,1.04])
plt.show()

#define the importance sampling method
def generate_1(N):
    total = 0
    for i in range(N):
        x = ra.normal(5,1)
        total += fx(x)/w(x)
    return total/N

#prepare the set for containing result
result_1 = np.zeros(100)

#combine the results into one array
for i in range(100):
    result_1[i] = generate_1(N)

#output the figure of the importance sampling method
plt.figure(2)  
plt.hist(result_1,10)
plt.show()


