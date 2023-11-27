# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 12:51:51 2023

@author: Rache
"""
import scipy.optimize as opt
import math
import numpy as np
kernel_size = 18
def Gaussian_2D(xy, x0, y0, amp_c, sigma_c):
    x,y = xy
    x0 = float(x0)
    y0 = float(y0)
    e = math.e
    center = amp_c*e**-(((x - x0)**2/sigma_c) + ((y-y0)**2/sigma_c))
    #surround = amp_s*e**-(((x - x0)**2/sigma_s) + ((y-y0)**2/sigma_s))
    #dog = center + surround 
    return center.ravel()

initial_guess = (kernel_size/2,kernel_size/2,1,1)
x = np.linspace(0,kernel_size-1,kernel_size)
y = np.linspace(0,kernel_size-1,kernel_size)
#x = np.linspace(-(kernel_size/2 + 1),kernel_size/2 + 1,kernel_size+1)
x, y = np.meshgrid(x,y)
test, test2 = opt.curve_fit(Gaussian_2D,(x,y), abs(lms_weights[0,:,:,0].ravel()), p0 = initial_guess)