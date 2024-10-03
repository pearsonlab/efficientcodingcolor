#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:25:44 2024

@author: david
"""

import numpy as np
import scipy

kernel_size = 18
n_lags = 10
n_colors = 2
#Create Cx from random distribution:
#Assumption: Here the spatial dimensions are distances in x and y 
Cx = abs(np.random.normal(size = [kernel_size, kernel_size, n_lags, n_colors, n_colors]))

#Make 2x2 Cx symmetric in colors, to solve generalized eigenvalue problem:
Cx[:,:,:,0,1] = Cx[:,:,:,1,0] 

#Fourier transform of Cx over spatial x spatial x temporal dimensions:
Cx_fft = np.fft.fftn(Cx, axes = [0,1,2])

eigvals = np.zeros([kernel_size, kernel_size, n_lags, n_colors])
eigvects = np.zeros([kernel_size, kernel_size, n_lags, n_colors, n_colors])
#Solve generalized eigenvalue problem of 
for x_freq in range(kernel_size):
    for y_freq in range(kernel_size):
        for temp_freq in range(n_lags):
            Cx_color = Cx_fft[x_freq,y_freq,temp_freq,:,:]
            eig = scipy.linalg.eigh(Cx_color)
            eigvals[x_freq,y_freq,temp_freq,:] = eig[0]
            eigvects[x_freq, y_freq, temp_freq, :,:] = eig[1]
            
            
#Step 2: Pick a temporal frequency
omega = 5

#Step 3: Create function to get a_c
def optim_filter(eig, nu, output_noise):
    left_side = (eig/eig+1)/2
    right_side = np.sqrt(1 + 4/(eig*nu*output_noise**2) + 1)
    output = output_noise**2*(left_side*right_side - 1)
    return np.clip(output, 0, np.inf)

#Step 4: 
alpha = 1.3
A = 100
input_noise = 0.4
k_f = (A/input_noise**2 * omega**2)**(1/alpha)


def k_tilda(k):
    return (k/k_f)**alpha




           
            