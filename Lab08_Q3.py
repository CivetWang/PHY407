# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 12:09:39 2018

@author: Civet
"""

from numpy import empty,zeros,max,exp
from pylab import imshow,gray,show
import numpy as np
import matplotlib.pyplot as plt

delta_x=0.02
delta_t=0.005
epsilon=1
Lx=2*np.pi
Tf=2
beta=epsilon*delta_t/delta_x
