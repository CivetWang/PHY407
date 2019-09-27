# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 17:01:31 2018

@author: lihaow
"""
#1. Re-import numpy & matplotlib & time 
import numpy as np
import matplotlib.pyplot as plt 
from time import time
#2. Set a certain matricx by modifiable dimension(from 2 to 100 )
mstart=2
mend=100
matrix = np.linspace(mstart,mend,mend-mstart+1)
#3. prepare the result entry for all running time
result = np.zeros(len(matrix))
#4. run all matrix multiplication and produce the array of running time
for i in range(len(matrix)-1):
    N = int(matrix[i])
    A = np.ones([N,N], float)*3
    B = np.ones([N,N], float)*3      
    C = np.ones([N,N], float)
    start = time()
    for i in range(N): 
        for j in range(N):
            for k in range(N):
                C[i,j]+=A[i,k]*B[k,j]
    end = time()
    result[i] = end-start
#5. set another array for result of np.dot
dot_time = np.zeros(len(matrix))
#6.run again with all matrix with np.dot
for i in range(len(matrix)-1):
    N = int(matrix[i])
    A = np.ones([N,N], float)*3
    B = np.ones([N,N], float)*3      
    start = time()
    C = np.dot(A,B)
    end = time()
    dot_time[i] = end-start
    
#7. Print times as functions of N
plt.figure(1)
plt.plot(matrix, result,label='our multiplication method')
plt.plot(matrix,dot_time,label='numpy.dot')
plt.xlabel('N')
plt.ylabel('running time')
plt.title('times as functions of N')
plt.legend()

#8. Print times as functions of N^3
plt.figure(2)
plt.plot(matrix**3, result,label='our multiplication method')
plt.plot(matrix**3,dot_time,label='numpy.dot')
plt.xlabel('N^3')
plt.ylabel('running time')
plt.title('times as functions of N^3')
plt.legend()


    