# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#packages for use
import math
import numpy as np

dim = 10#dimension of the hypersphere
r = 1#radius of the hypersphere
N = 1000000#runtime target
V = (2*r)**dim#volume boundary region

#define the real volume 
def real(R,n):
    return (R**n)*(np.pi**(n/2))/(math.gamma(n/2 + 1))

#define the value justifying function
def fxy(array):
    norm = np.linalg.norm(array)
    if norm <=1:
        c = 1
    else: 
        c = 0
    return c

#define random generating function for each integral region
def generate (N):
    total = 0
    for i in range(N):
        ten_dim = np.random.random(10)*2-1
        total += fxy(ten_dim)
    return total

result = V/N*generate(N)#compute I with Eq 10.33
error  = np.sqrt(result*(2**10-result)/N)#calculate error qith Eq 10.26

#output           
print('The calculated value is', result)
print('The real value is', real(r,dim))
print('The error of the calculated value is',error)