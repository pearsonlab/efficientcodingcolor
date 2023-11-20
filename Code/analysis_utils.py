# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 22:45:40 2023

@author: David
"""
import numpy as np
import os
import torch

def find_last_cp(path):
    all_files = np.array(os.listdir(path))
    max_cp = 0
    for file in all_files:
        if file[0:10] == 'checkpoint':
            cp_num = int(file[11:-3])
            if cp_num > max_cp:
                max_cp = cp_num
                max_cp_file = file
    return max_cp_file

def find_last_model(path):
    all_files = np.array(os.listdir(path))
    max_model = 0
    for file in all_files:
        if file[0:5] == 'model':
            model_num = int(file[6:-3])
            if model_num > max_model:
                max_model = model_num
                max_model_file = file
    return max_model_file

def reshape_flat_W(W, n_neurons, kernel_size, n_colors):
    W_pre = W.reshape([n_colors,kernel_size,kernel_size,n_neurons])
    W_reshape = np.swapaxes(W_pre,0,3)
    return W_reshape

def round_kernel_centers(kernel_centers):
    kernel_centers_int = np.zeros(kernel_centers.shape, dtype = np.int16)
    n_neurons = kernel_centers.shape[0]
    for n in range(n_neurons):
        kernel_centers_int[n,0] = int(round(kernel_centers[n,0]))
        kernel_centers_int[n,1] = int(round(kernel_centers[n,1]))
    return kernel_centers_int

def scale(W, W_all = None):
    if W_all is not None:
        W_scale = W_all
    else:
        W_scale = W
    
    
    W_max = np.max(W_scale)
    W_min = np.min(W_scale)
    if abs(W_max) < abs(W_min):
        ext = abs(W_min)
    else:
        ext = abs(W_max)
    
    W = W/(2*ext) + 0.5
    return W

def get_matrix(matrix_type):
    pca_comps_old = np.array([[ 0.51876956, 0.52288215, 0.67636706],
                 [ 0.48552343, 0.4709887, -0.73650299],
                 [ 0.7036655, -0.71046739, 0.00953693]])
    
    pca_comps = np.array([[ 0.56808728,  0.57183637,  0.59184459],
           [ 0.42497522,  0.41201491, -0.80600234],
           [ 0.70475024, -0.70939896,  0.00895586]])
    pca_inv = np.linalg.inv(pca_comps)  
    
    #Do you see what I see? -Understanding the challenges of colour-blindness in online learning
    rgb_to_lms = np.array([[17.88240413, 43.51609057,  4.11934969],
                           [ 3.45564232, 27.15538246,  3.86713084],
                           [ 0.02995656,  0.18430896,  1.46708614]])
    lms_to_rgb = np.linalg.inv(rgb_to_lms)
    
    if matrix_type == 'pca_comps':
        return pca_comps
    elif matrix_type == 'pca_inv':
        return pca_inv
    elif matrix_type == 'rgb_to_lms':
        return rgb_to_lms
    elif matrix_type == 'lms_to_rgb':
        return lms_to_rgb
    elif matrix_type == 'pca_comps_old':
        return pca_comps_old
    
def params_preprocess(all_params):
    n_params = all_params.shape[0]
    a_index = np.array(range(0,n_params,4))
    b_index = a_index + 1
    c_index = a_index + 2
    d_index = a_index + 3
    a_pre = all_params[a_index,:].exp()
    b = all_params[b_index,:].exp()
    a = a_pre + b
    c = torch.sigmoid(all_params[c_index,:])
    d = all_params[d_index,:]/torch.sqrt(torch.sum(all_params[d_index,:]**2, 0))
    return a, b, c, d

def make_rr_v2(og_size, new_size):
    x = torch.arange(new_size)
    y = torch.arange(new_size)
    grid_x, grid_y = torch.meshgrid(x, y)

def make_rr(og_size, new_size, kernel_center = [0,0]):
    if new_size%2 !=0:
        Exception("New size must be even!")
    middle = og_size/2
    x,y = np.linspace(-middle, middle - 1, new_size), np.linspace(-middle, middle - 1, new_size)
    rr = np.meshgrid(x,y)
    rr = np.expand_dims(rr, 0)
    return rr 
   

def closest_divisor(number):
    min_add = np.inf
    n_final = 0
    m_final = 0
    for n in range(1, number):
        if number%n == 0:
            m = number/n
            if n + m < min_add:
                min_add = n + m
                n_final = n
                m_final = int(m)
    return n_final, m_final
    
                
            
    
