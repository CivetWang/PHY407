# -*- coding: utf-8 -*-
"""
Spyder Editor
@author: Civet
"""

#Pseudocode:
#1.First of first, import numpy
#2. Secondly, import speed of light data
#3. Then Write the two methods into functions
#4. Compute the relative errors:
    #calculate the correct answer
    #calculate std by method1
    #calculate std by method2
    #calculate and print the relative errors respect to the correct answer
    
    
#1.First of first, import numpy
import numpy as np

#2. Secondly, import speed of light data
cdata = np.loadtxt('cdata.txt')

#3. Then Write the two methods into functions
def method1(data):
    n = len(data)
    mean = np.average(data)
    std = np.sqrt(1./(n-1)*sum((data - mean)**2))
    return std

def method2(data):
    n = len(data)
    mean = np.average(data)
    std = np.sqrt(1./(n-1)*(sum(data**2)-n*mean**2))
    return std



#4. Compute the relative errors
def relative_error(data):
   std = np.std(data,ddof=1) #calculate the correct answer
   std_1 = method1(data) #calculate std by method1
   std_2 = method2(data) #calculate std by method2
   #calculate and print the relative errors respect to the correct answer
   err_1 = (std_1-std)/std
   err_2 = (std_2-std)/std
   print ('relative error from Eqn 1 is ',err_1)
   print ('relative error from Eqn 2 is ',err_2)
   return [err_1,err_2]

#4. Output the result for data
print('For Michelsen’s speed of light data')
relative_error(cdata)

#Question (C) 
#5. Create sequences then run through the same process for standard deviation,
#   such that the ralative error for each method is computable
random_1 = np.random.normal(0., 1., 2000)
random_2 = np.random.normal(1.e7, 1., 2000)
print('For random with mean 0.')
relative_error(random_1)
print('For random with mean 1.e7')
relative_error(random_2)

test_size = 1000

result_11 = np.zeros(test_size)
result_12 = np.zeros(test_size)
result_21 = np.zeros(test_size)
result_22 = np.zeros(test_size)

#compute the relative results
for i in range(test_size):
    random_1 = np.random.normal(0., 1., 2000)
    random_2 = np.random.normal(1.e7, 1., 2000)
    result_11[i] = relative_error(random_1)[0]
    result_12[i] = relative_error(random_1)[1]
    result_21[i] = relative_error(random_2)[0]
    result_22[i] = relative_error(random_2)[1]

import matplotlib.pyplot as plt
#plot the order of magnitude of relative errors
plt.plot(range(test_size),np.log10(np.abs(result_11)),':',label='relative error of sequences with 0 mean by eqn1')
plt.plot(range(test_size),np.log10(np.abs(result_12)),':',label='relative error of sequences with 0 mean by eqn2')
plt.plot(range(test_size),np.log10(np.abs(result_21)),'y:',label='relative error of sequences with 1e7 mean by eqn1')
plt.plot(range(test_size),np.log10(np.abs(result_22)),':',label='relative error of sequences with 1e7 mean by eqn2')
plt.legend()
plt.ylabel('order of magnitude of relative errors')



