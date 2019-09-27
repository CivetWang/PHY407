# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 20:03:12 2018

@author: student
"""

from numpy import arange, pi, sin
from pylab import clf, plot, xlim, ylim, show, pause,draw
t = arange(0, 4*pi, pi/100)
# t coordinate
x = arange(0,4*pi, pi/100)
# x coordinate
for tval in t:
    clf()
    # clear the plot
    plot(x, sin(x-tval))
    # plot the current sin curve
    xlim([0, 4*pi])
    # set the x boundaries constant
    ylim([-1, 1])
    # and the y boundaries
    draw()
    pause(0.01)
    #pause to allow a smooth animation