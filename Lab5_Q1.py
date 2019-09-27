# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 16:02:21 2018

@author: student
"""
"""
Use Fourier Transform to analysis different set of data and find periodicity
of thae target which can  
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft,irfft
from numpy import empty,arange,exp,real,pi

# define discrere fourier transform
def dft(y):
    N = len(y)
    c = np.zeros(N//2+1,complex)
    for k in range(N//2+1):
        for n in range(N):
            c[k] += y[n]*exp(-2j*pi*k*n/N)
    return c
# define fast cosine transform
def dct(y):
    N = len(y)
    y2 = empty(2*N,float)
    y2[:N] = y[:]
    y2[N:] = y[::-1]

    c = rfft(y2)
    phi = exp(-1j*pi*arange(N)/(2*N))
    return real(phi*c[:N])
# define inverse fast cosine transform                
def idct(a):
    N = len(a)
    c = empty(N+1,complex)

    phi = exp(1j*pi*arange(N)/(2*N))
    c[:N] = phi*a
    c[N] = 0.0
    return irfft(c)[:N]

#Qa
y=np.loadtxt("sunspots.txt")#load data
x=y[:,1]
x-=np.mean(x)
plt.figure(1)
plt.plot(x)
plt.title('Sunspots respect to month')
plt.ylabel('sunspot number')
plt.xlabel('time/month')
plt.show()
ss=dft(x)# apply discrere fourier transform
ss1=abs(ss)**2
plt.figure(2)
plt.plot(ss1)
plt.title('Coefficient Diagram')
plt.ylabel('Square of Absolute value of Ck')
plt.xlabel('k')
plt.show()
plt.figure(3)
plt.plot(ss1[0:50])
plt.title('Peak Zoom')
plt.ylabel('Square of Absolute value of Ck')
plt.xlabel('k')
plt.show()

#Qb
dj=np.loadtxt("dow.txt")#load data
plt.figure(4)
dj-=np.mean(dj)
plt.plot(dj,label='Index')
plt.title('Dow Jones chart 1 from late 2006 to end 2010 for first 2%')
plt.ylabel('Index/Ck')
plt.xlabel('Time(day)/k')
ftdj=rfft(dj)# apply fast fourier transform
ftdj_m=ftdj
ftdj_m[int(0.02*len(ftdj_m)):]=0       
#apply inverse fast fourier transform to rebuild the curve
plt.plot(irfft(ftdj_m),label='inverse Fourier Trans')
plt.legend()
plt.show()

#Qc
dj2=np.loadtxt("dow2.txt")#load data
plt.figure(5)
dj2-=np.mean(dj2)
plt.plot(dj2,label='Index')
plt.title('Dow Jones chart 2 from 2004 to 2008[with rfft method]')
plt.ylabel('Index/Ck')
plt.xlabel('Time(day)/k')
ftdj2=rfft(dj2)# apply fast fourier transform
ftdj_m2=ftdj2
ftdj_m2[int(0.02*len(ftdj_m2)):]=0   
#apply inverse fast fourier transform to rebuild the curve
plt.plot(irfft(ftdj_m2),label='inverse Fourier Trans[RFFT]')
ftdj2=dct(dj2)# apply fast cosine transform
ftdj_m2=ftdj2
ftdj_m2[int(0.02*len(ftdj_m2)):]=0   
#apply inverse fast fourier transform to rebuild the curve
plt.plot(idct(ftdj_m2),label='inverse Fourier Trans[DCT]')
plt.legend()
plt.show()

#Qd
pin=np.loadtxt("piano.txt")#load data
plt.figure(7)
plt.plot(pin)
plt.title('Waveform of Piano')
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.show()
cof=rfft(pin)# apply fast fourier transform
cof1=abs(cof)
plt.figure(8)
plt.plot(cof1[:10000])
plt.title('Coefficient Diagram')
plt.ylabel('Ck')
plt.xlabel('k')
plt.show()

tru=np.loadtxt("trumpet.txt")#load data
plt.figure(9)
plt.plot(tru)
plt.title('Waveform of  Trumpet')
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.show()
coff=rfft(tru)# apply fast fourier transform
coff1=abs(coff)
plt.figure(10)
plt.plot(coff1[:10000])
plt.title('Coefficient Diagram')
plt.ylabel('Ck')
plt.xlabel('k')
plt.show()