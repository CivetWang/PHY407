# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 10:53:03 2018

@author: Civet
"""
#Import what packages like numpy matplotlib as usual, for specific functions,
# may import seperatly, for a better common use frequency
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from numpy.fft import  fft2,ifftshift,fftshift

#import a sample file image, Which I used one I picked from my own collection,
#which can be change to any file name or change to a input variable to make it 
#more widly available to use (as the method 17,but I won't use it this way this
# time)
#name = input("File name =")
name = 'IMG001.jpg'#input("File name =")
img = img.imread(name)
R = img[:,:,0]
G = img[:,:,1]
B = img[:,:,2]
#save all three RGB layers into certainer for possible future usage


#setting up the filter matrice as what has been written in the report: 
#construct them with zeros and ones, the constants used below is the size of 
#the window size in each filter.
N_L=160
N_H=80
rows,cols = R.shape
#High pass filter
mask = np.ones(R.shape,np.uint8)
mask[int(rows/2)-N_H:int(rows/2)+N_H,int(cols/2)-N_H:int(cols/2)+N_H] = 0
#Low pass filter
mask_1 = np.zeros(R.shape,np.uint8)
mask_1[int(rows/2)-N_L:int(rows/2)+N_L,int(cols/2)-N_L:int(cols/2)+N_L] = 1
#---------------------------------------------
#define RGB diagram to gray scale diagram
#as the packages imported don't have a gray scale transform function, a scale 
#of RGB ratio must be manually input. As a easy scale of 1/3 of each layer, a 
# more accurate ratio of 0.299 * R + 0.587 * G + 0.114 * B was found online,
# with a better clarity, this ratio was chosen to be keep.
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# define the filter-Fourier transformation compression method
# As steps are vary similar for the three basic filters, I compress all process 
# that repeats times by time into a funtion to make the script efficient and 
# clearer to read .
def compress(image,N,mask):
    phase1 = fft2(image)# transform into frequency domain
    phase1shift = fftshift(phase1)# centerolize the forurier result
    phase2shift = ifftshift(phase1shift*mask) 
    #filter it and decenterolize the function
    img_new = np.abs(np.fft.ifft2(phase2shift))
    # take the real part off the result and keep in usage
    img_new = (img_new-np.amin(img_new))/(np.amax(img_new)-np.amin(img_new))
    #respread the outpiut through the range
    return img_new
#---------------------------------------------
#image compressing in gray scale with all three methods to compare with each 
# other for a analysis of each result output.
gray = rgb2gray(img)        
plt.subplot(231)
plt.imshow(compress(gray,N_H,mask),'gray')
plt.title('High-pass-Gray')
plt.subplot(232)
plt.imshow(compress(gray,N_L,mask_1),'gray')
plt.title('Low-pass-Gray')
plt.subplot(233)
plt.imshow(compress(gray,N_H,mask*mask_1),'gray')
plt.title('Band-pass-Gray')

#---------------------------------------------
# After gray scale, proceed the same process to the RGB image, however, as RGB
# has three layers, all three should be through the same filter in frequency 
# domain.
# Low pass
r=compress(R,50,mask_1)*255
g=compress(G,50,mask_1)*255
b=compress(B,50,mask_1)*255
# Create the result container
result=np.zeros([len(img),len(img[0]),3],dtype=np.uint8)
# fill the container with results
result[:,:,0]=r
result[:,:,1]=g
result[:,:,2]=b
plt.subplot(234)
plt.imshow(result)
plt.title('Low_pass-RGB')
# High pass
r=compress(R,50,mask)*255
g=compress(G,50,mask)*255
b=compress(B,50,mask)*255
result[:,:,0]=r
result[:,:,1]=g
result[:,:,2]=b
plt.subplot(235)
plt.imshow(result)
plt.title('High-pass-RGB')
# Band pass
r=compress(R,50,mask*mask_1)*255
g=compress(G,50,mask*mask_1)*255
b=compress(B,50,mask*mask_1)*255
result[:,:,0]=r
result[:,:,1]=g
result[:,:,2]=b
plt.subplot(236)
plt.imshow(result)
plt.title('Band-pass-RGB')
#Last output the Original Diagram for a visual caomparison
plt.figure(2)
plt.imshow(img)
plt.title(' Original')
# As a side support I did the plot for the filter sample,although the basic 
# three filter looks the same visually in such a big scale, they are pretty 
# different while dimensions are not that huge.
plt.figure(3)
f = np.fft.fft2(mask_1)
f1shift = np.fft.fftshift(f)
img = np.log(np.abs(f1shift))
plt.imshow(img,'gray')
plt.title(' Low-pass filter in this project’s size')
# This is a sample of Low pass filter in frequency domain I copyed from 
# https://blog.csdn.net/on2way/article/details/46981825
# which is where most support from online, which most lead my thought and 
#  progress through the hole project.
plt.figure(4)
laplacian = np.array([[0,1,0],[1,-4,1],[0,1,0]])
f = np.fft.fft2(laplacian)
f1shift = np.fft.fftshift(f)
img = np.log(np.abs(f1shift))
plt.imshow(img,'gray')
plt.title(' Low pass filter in a 3x3 scale')

# Thanks again for reading this script to the end(I hope someone did), thanks 
# again for such a great course I take this year, I know there's still several 
# filter out there, for example, Guassian Filter, Mean Filter, Median Filter 
# and so on, but I choose to not proceed too far this time. As the most target 
# has already completed among the progress.However, I found that finding out 
# these filter and developing something by oneself is really something enjoyable.
# Just as a suggestion, probably that is a good assignment for next year student
# as some of the student might from engineer or study signal analysis, that's 
# really helpful.Thanks again for being such a great course now.orz
