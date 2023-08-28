# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 16:32:34 2023

@author: Rache
"""
import numpy as np
import matplotlib.pyplot as plt
from analysis_utils import closest_divisor
import time
n_neurons = 100
kernel_size = 12

def hexagonal_grid(n_neurons, radius, kernel_size):
    dist = 1
    goal_neurons = False
    while not goal_neurons:
        size = int(kernel_size/2)
        #n_x, n_y = closest_divisor(n_neurons)
        x_all = np.arange(-size,size,dist)
        y_all = np.arange(-size,size,dist)
        
        x, y = np.meshgrid(x_all, y_all)
        for x_pos in range(x.shape[1]):
            if x_pos%2 == 0:
                x[x_pos,:] = x[x_pos,:] - dist/2
        
        x_flat = x.flatten()
        y_flat = y.flatten()
        kernel_centers = np.stack((x_flat,y_flat),1)
    
        center_dist = np.sqrt(x_flat**2 + y_flat**2)
        within_radius = np.sum(center_dist < radius)
        if within_radius < n_neurons:
            dist = dist - 0.001
        elif within_radius > n_neurons:
            dist = dist + 0.001
        else:
            goal_neurons = True
        
    kernel_centers = kernel_centers[center_dist < radius, :]
    return kernel_centers