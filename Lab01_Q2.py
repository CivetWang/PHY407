# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 17:01:47 2018

@author: hengl
"""
#1. import the packages we might use
import numpy as np
from numpy.random import random
import matplotlib.pyplot as plt

#2. write the equation in Q2a into a function
def scattering(z):
    return (np.pi - 2*np.arcsin(z))/np.pi*180


#3. set the random height series z
z = (random(1000000)-0.5)*2

#4. compute the scattering angles theta
theta = scattering(z)

#5. plot the histogram of the incoming heights
plt.figure(1)
plt.hist(theta,300)
plt.xlabel('θ/degree')
plt.ylabel('counts')
plt.title('histogram of outcoming angle')

#6. plot the histogram of the scattering angles
plt.figure(2)
plt.hist(z,360)
plt.xlabel('height')
plt.ylabel('counts')
plt.title('histogram of incoming height')

#7. calculate the relative probability
prob1 = float(((170<theta) & (theta<190)).sum()) / len(theta)
prob2 = float(((90<theta) & (theta<110)).sum()) / len(theta)
reprob = prob1 / prob2
print(reprob)