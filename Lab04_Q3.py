# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 09:39:06 2018

@author: Civet
"""
from math import exp
from numpy import linspace,sign
from pylab import plot,clf,show
import matplotlib.pyplot as plt
#Q3(a)
x1=0.01
accuracy = 1e-6
error = 1
while error > accuracy:
    x1,x2= 1-exp(-2*x1),x1
    error=abs((x1-x2)/(1-1/(2*exp(-2*x1))))
print(x1)

clf()
max=3.01
accuracy = 1e-6
points=300
y=list()
line = linspace(0.01,max,points)
for c in line:
    x1=1
    error = 1
    while error > accuracy:
        x1,x2= 1-exp(-c*x1),x1
        error=abs((x1-x2)/(1-1/(c*exp(-c*x1))))
    y.append(x1)
plot(line,y) 
plt.ylabel('x value')
plt.xlabel('c value')
show()
#Q3(b)        
x1=0.01
accuracy = 1e-6
error = 1
c=0
while error > accuracy:
    x1,x2= 1-exp(-2*x1),x1
    error=abs((x1-x2)/(1-1/(2*exp(-2*x1))))
    c+=1
print(c)

x1=0.01
accuracy = 1e-6
error = 1
c=0
w=1.1
while error > accuracy:
    x1,x2= (1+w)*(1-exp(-2*x1))-w*x1,x1
    error=abs((x1-x2)/(1-1/(2*exp(-2*x1))))
    c+=1
print(c)
print(x1)
#Q3(c)
def f(x):
    return 5-5*exp(-x)
# Relaxation method
x1=0.01
accuracy = 1e-6
error = 1
c=0
while error > accuracy:
    x1,x2= f(x1),x1
    error=abs((x1-x2)/(1-1/(5*exp(x1))))
    c+=1
print(c)
print(x1)

def f(x):
    return 5*exp(-x)+x-5
def f_p(x):
    return -5*exp(-x)+1

#Newton's method
x1=7
accuracy = 1e-6
error = 1
c=0
while error > accuracy:
    x1,x2= x1-f(x1)/f_p(x1),x1
    error=-f(x1)/f_p(x1)
    c+=1
print(c)
print(x1)
# Binary search
x1=4
x2=7
accuracy = 1e-6
error = 1
c=0

while error > accuracy:
    x=(x1+x2)*0.5
    if sign(f(x))==sign(f(x1)):
        x1=x
    elif sign(f(x))==sign(f(x2)):
        x2=x
    error=abs(x1-x2)
    c+=1
print(c)
print(x1)

Pc=6.62607004e-34 # redused Planck Constant
kB = 1.38064852e-23# boltzmann constant
c = 299792458# speed of light
b=(Pc*c)/(kB*x1)
lam=5.02e-7
print(b/lam)