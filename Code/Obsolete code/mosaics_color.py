# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 19:45:54 2022

@author: David
"""
import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import torch
from util import cycle, kernel_images, get_matrix, scale_0to1
from PIL import Image
from numpy.linalg import norm
import pandas as pd
import scipy.optimize as opt
import math
matplotlib.use('QtAgg')

#Model you want to do the mosaic from
#save = '221009-000434'
#save = '230403-013925'
#save ='230404-121120'
#save = '230409-231107'
#save = '230413-234143'
#save = '230418-005023'
save = '230421-012342'
#save = '230502-125114'
#save = '230508-014147'
save = '230619-020142'
save = '230620-014248'

path = "saves/" + save + "/"
#Open log file to figure out if this simulation had PCA pre-processing or not
logs = pd.read_csv('./Simulation_logs.csv')
#pca_preprocess = logs['PCA'][logs['ID'] == save].values[0] == 'TRUE'
pca_preprocess = False
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
            



#Import learned model and its weights. Figure out n_colors, kernel_size and output_neurons from weights size. 
cp = torch.load(path + find_last_cp(path)) #bug here plz fix
kernel_size = cp['args']['kernel_size']
#weights = cp['model_state_dict']['encoder.W'].cpu().detach().numpy()
weights = cp['weights'].cpu().detach().numpy()
output_neurons = weights.shape[1]
n_colors = int(weights.shape[0]/(kernel_size**2))

#Reshape weights and obtain pca_inv and lms_to_rgb matrices.
weights = np.reshape(weights, [n_colors,kernel_size,kernel_size,output_neurons])
pca_inv = get_matrix(matrix_type = 'pca_inv')
lms_to_rgb = get_matrix(matrix_type = 'lms_to_rgb')

#Transform weights in LMS and RGB space
if pca_preprocess:
    lms_weights = np.tensordot(pca_inv, weights, axes = 1)
else:
    lms_weights = weights
rgb_weights = np.tensordot(lms_to_rgb, lms_weights, axes = 1)

#Reshape rgb and lms weights so n_colors is last to use plt.imshow
rgb_weights = np.swapaxes(rgb_weights, 0, 3)

lms_weights = np.swapaxes(lms_weights, 0, 3)
#Plots all RFs per color. Expects weights of size [n_colors, kernel_size, kernel_size, n_neurons]
def my_kernel_images(weights, normalize = True):
    n_colors = weights.shape[0]
    n_neurons = weights.shape[3]
    size = weights.shape[1]
    sqrt_n = int(np.sqrt(n_neurons))
    if n_neurons == sqrt_n**2:
        add_row = 0
    else:
        add_row = 1
    kernels = np.zeros([weights.shape[0], weights.shape[1]*sqrt_n, weights.shape[2]*(sqrt_n+add_row)])
    for color in range(n_colors):
        for n in range(n_neurons):
            y_pos = (int(n/sqrt_n))
            x_pos = (n - y_pos*sqrt_n)
            x_pos = x_pos*size; y_pos = y_pos*size
            this_kernel = weights[color,:,:,n]
            if normalize:
                this_kernel = this_kernel/norm(this_kernel)
            kernels[color,x_pos:x_pos+size,y_pos:y_pos+size] = this_kernel
            
    return kernels

#Separates list of neurons into L-center, M-center and S-center neurons based on the maximum pixel value. 
#expects input of size [n_colors, kernel_size, kernel_size, n_neurons]
def select_lms(W):
    n_neurons = W.shape[3]
    kernel_size = W.shape[2]
    L_neurons, M_neurons, S_neurons = np.zeros([1, kernel_size, kernel_size]), np.zeros([1, kernel_size, kernel_size]), np.zeros([1, kernel_size, kernel_size])
    for neuron in range(n_neurons):
        Wi = W[:,:,:,neuron]
        max_index = np.where(Wi == np.max(Wi))
        max_color = max_index[0][0]
        Wi_to_add = np.expand_dims(Wi[max_color,:,:], axis = 0)
        if max_color == 0:
            L_neurons = np.append(L_neurons, Wi_to_add, axis = 0)
        if max_color == 1:
            M_neurons = np.append(M_neurons, Wi_to_add, axis = 0)
        if max_color == 2:
            S_neurons = np.append(S_neurons, Wi_to_add, axis = 0)
    L_neurons = L_neurons[1:,:,:]
    M_neurons = M_neurons[1:,:,:]
    S_neurons = S_neurons[1:,:,:]
    return L_neurons, M_neurons, S_neurons

def select_on_off2(W, all_c):
    neurons_types = on_or_off(W, all_c)
    on_neurons = W[neurons_types,:,:]
    off_neurons = W[np.invert(neurons_types),:,:]
    return on_neurons, off_neurons
    
#expcets input of size [n_neurons, kernel_size, kernel_size, n_colors]
#divides W into two matrices, one for ON-center neurons and one for OFF-center neurons
def select_on_off(W):
    W_sum = np.sum(W, axis = 3)
    neurons_types = return_sign(W_sum)
    on_neurons = W[neurons_types,:,:]
    off_neurons = W[np.invert(neurons_types),:,:]
    return on_neurons, off_neurons

#Returns an array that tells whether each neuron is ON or OFF-center. Does so by looking at maximum values
#Expects an array of shape [n_neurons, kernel_size, kernel_size] or [n_neurons, kernel_size, kernel_size, n_colors]
def return_sign(W):
    sign = abs(np.min(np.min(W, axis = 1), axis = 1)) < np.max(np.max(W, axis = 1), axis = 1)
    return sign


#Does a contour plot for each neuron and return its contour line coordinates
#Takes as input weights of shape []
def contour_coord(W):
    n_neurons = W.shape[0]
    
    W = np.sum(W, axis = 3)
    ON = return_sign(W)
    sign_mul = 2*(ON.astype(int) - 0.5)
    all_c = np.zeros([2, n_neurons])
    for n in range(n_neurons):
        fig = plt.contour(sign_mul[n]*W[n,:,:], levels = [-0.1,0.1])
        c = center_of_mass(fig.allsegs[-1][0])
        all_c[:,n] = [c[0], c[1]]
    plt.close()
        
    return all_c

def fit_Gaussian_2D(xy, x0, y0, amp_c, sigma_c):
    x,y = xy
    x0 = float(x0)
    y0 = float(y0)
    e = math.e
    center = amp_c*e**-(((x - x0)**2/sigma_c) + ((y-y0)**2/sigma_c))
    #surround = amp_s*e**-(((x - x0)**2/sigma_s) + ((y-y0)**2/sigma_s))
    #dog = center + surround 
    return center.ravel()

def contour_coord_gauss(W):
    n_neurons = W.shape[0]; kernel_size = W.shape[1]
    all_c = np.zeros([2, n_neurons])
    W = np.mean(abs(W), 3)
    #W = abs(np.mean(W,3))
    initial_guess = (kernel_size/2,kernel_size/2,1,1)
    x = np.linspace(0,kernel_size-1,kernel_size)
    y = np.linspace(0,kernel_size-1,kernel_size)
    x, y = np.meshgrid(x,y)
    
    for n in range(n_neurons):
        params, cov = opt.curve_fit(fit_Gaussian_2D, (x,y), W[n,:,:].ravel(), p0 = initial_guess)
        all_c[:,n] = params[0:2]
    return all_c
    
def c_nearest_neighbor(all_c):
    n_neurons = all_c.shape[0]
    c_nearest = np.zeros(n_neurons)
    for n1 in range(n_neurons):
        nearest = np.inf
        for n2 in range(n_neurons):
            dist = ((all_c[n2,0] - all_c[n1,0])**2 + (all_c[n2,1] - all_c[n1,1])**2)**0.5
            if dist < nearest and n1 != n2:
                c_nearest[n1] = dist
                nearest = dist
    nearest_min = np.sort(c_nearest)
    nearest_pairs = []
    for m in range(n_neurons):
        pair = [i for i, n in enumerate(c_nearest) if n == np.min(nearest_min[m])]
        nearest_pairs.append(pair)
    return nearest_pairs

#Finds the center of mass of a neuron based on the coordinates of its contour plot
def center_of_mass(X):
    x_mean = np.mean(X[:,0])
    y_mean = np.mean(X[:,1])
    return [x_mean,y_mean]

#Makes mosaic from center of mass coordinates
def make_mosaic(weights, all_c, marker):
    #c = contour_coord_gauss(weights)
    #plt.figure()
    ax = plt.gca()
    ax.set_aspect('equal')
    for n in range(all_c.shape[1]):
        ax.plot(all_c[0,n], all_c[1,n], marker = marker, markersize = 6, color = 'black')
    ax.set_title('Mosaic of RF centers')

#Makes both mosaics for ON and OFF center neurons
def make_mosaics(weights, all_c):
    W1, W2 = select_on_off2(weights, all_c) 
    is_on = on_or_off(weights, all_c)
    all_c_on = all_c[:,is_on]
    all_c_off = all_c[:, np.invert(is_on)]
    plt.figure()
    make_mosaic(W1, all_c_on, 'o')
    make_mosaic(W2, all_c_off, 'x')

#Create a [n_neurons, n_colors] matrix of the color strength at each neuron's RF center. 
def return_center_colors(weights, all_c, normalize_sd = False, remove_mean = False):
    n_neurons = weights.shape[0]
    n_colors = weights.shape[3]
    center_color = np.zeros([n_neurons, n_colors])
    for n in range(n_neurons):
        center_color[n,:] = weights[n, round(all_c[0,n]), round(all_c[1,n]), :]
        if remove_mean:
            center_color[n,:] = center_color[n,:] - np.mean(center_color[n,:])
        if normalize_sd:
            center_color[n,:] = (center_color[n,:])/(np.sum(abs(center_color[n,:])))
    return center_color

#Computes average LMS responses as a function of distance from RF center. Only looks a pixels direclty up, down, left or right of the center. 
#Takes as input a neuron's weights [kernel_size, kernel_size, n_colors] and the x-y coordinates of its center (integer) 
def radial_average(neuron_weights, c, rad_range = 10):
    n_colors = neuron_weights.shape[2]
    kernel_size = neuron_weights.shape[1]
    y = round(c[0])
    x = round(c[1])
    rad_avg = np.zeros([rad_range,n_colors])
    
    for r in range(rad_range):
        tot = 4
        if y + r >= 0 and y + r <= kernel_size - 1:
            up = neuron_weights[x, y + r, :]
        else:
            tot = tot - 1; up = 0
        if y - r >= 0 and y - r <= kernel_size - 1:  
            down = neuron_weights[x, y - r, :]
        else:
            tot = tot - 1; down = 0
        if x + r >= 0 and x + r <= kernel_size - 1:
            right = neuron_weights[x + r, y, :]
        else:
            tot = tot - 1; right = 0
        if x - r >= 0 and x - r <= kernel_size - 1:
            left = neuron_weights[x - r, y, :]
        else:
            tot = tot - 1; left = 0
        if tot > 0:
            rad_avg[r,:] = (up + down + right + left)/tot
        else:
            rad_avg[r,:] = 0
    return rad_avg

def return_subplot(axes, n, n_neurons):
    size = int(np.sqrt(n_neurons))
    y_pos = int(n/size)
    x_pos = n - y_pos*size
    axis = axes[x_pos, y_pos]
    return axis

def round_all_c(all_c):
    all_c_int = np.zeros(all_c.shape, dtype = np.int16)
    n_neurons = all_c.shape[1]
    for n in range(n_neurons):
        all_c_int[0,n] = int(round(all_c[0,n]))
        all_c_int[1,n] = int(round(all_c[1,n]))
    return all_c_int

def on_or_off(W, all_c):
    W_sum = np.mean(W, axis = 0)
    all_c_int = round_all_c(all_c)
    n_neurons = W.shape[3]
    neurons_types = np.empty([n_neurons], dtype = np.bool_)
    for n in range(n_neurons):
        neurons_types[n] = W_sum[all_c_int[n, 1], all_c_int[n, 0], n] > 0
    return neurons_types



def plot_radial_averages(weights, c, rad_range = 10):
    n_neurons = weights.shape[0]
    size = int(np.sqrt(n_neurons))
    if n_neurons%size != 0:
        extra_column = 1
    else:
        extra_column = 0
    fig, axes = plt.subplots(size,size+extra_column)
    for n in range(n_neurons):
        y_pos = int(n/size)
        x_pos = n - y_pos*size,
        axis = axes[x_pos, y_pos][0]
        #axis = return_subplot(axes, n, n_neurons)
        rad_avg = radial_average(weights[n,:,:,:], c[:,n], rad_range = rad_range)
        axis.plot(rad_avg[:,0], color = 'r')
        axis.plot(rad_avg[:,1], color = 'g')
        axis.plot(rad_avg[:,2], color = 'b')
        axis.axhline(0, color = 'black')
        axis.set_title("# " + str(n), fontsize = 6)
    fig.suptitle("LMS weights as a function of distance from center", y = 0.92, size = 30)
    fig.supxlabel('Radial distance from center (pixels)', y = 0.05, size = 24)
    fig.supylabel('Weight', x = 0.1, size = 24)
        



kernels = my_kernel_images(weights)
if pca_preprocess:
    kernels_lms = np.tensordot(pca_inv, kernels, axes = 1)
else:
    kernels_lms = kernels
fig, [ax1, ax2, ax3] = plt.subplots(1, 3)

c = 0
lms = np.array(["L", "M", "S"])
for ax in [ax1,ax2,ax3]:
    ax.imshow(kernels_lms[c,:,:], cmap = 'gray')
    major_ticks = np.arange(0, 253, 18)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.grid()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(lms[c] + " cones")
    c = c + 1
all_c = contour_coord_gauss(lms_weights)

def plot_all_centers(weights, all_c):
    n_neurons = weights.shape[0]
    size = int(weights.shape[0]**0.5)
    if n_neurons%size != 0:
        extra_column = 1
    else:
        extra_column = 0
    fig, axes = plt.subplots(size,size+extra_column)
    
    weights = (weights - np.min(weights))/(np.max(weights) - np.min(weights))
    for n in range(weights.shape[0]):
        y_pos = int(n/size)
        x_pos = n - y_pos*size
        axis = axes[x_pos, y_pos]
        axis.imshow(weights[n, :, :, :], vmin = np.min(weights), vmax = np.max(weights))
        axis.plot(all_c[0,n], all_c[1,n], marker = 'x', markersize = 10, color = 'black')
        #print(lms_weights[n, round(all_c[0,n]), round(all_c[1,n]), :])

kernels_rgb = np.tensordot(lms_to_rgb, kernels_lms, axes = 1)
kernels_rgb = np.swapaxes(kernels_rgb, 0, 2); kernels_lms = np.swapaxes(kernels_lms,0,2)
kernels_rgb = scale_0to1(kernels_rgb)
kernels_lms = scale_0to1(kernels_lms)

fig, [ax1, ax2] = plt.subplots(1,2)
ax1.imshow(kernels_rgb)
ax2.imshow(kernels_lms)
ticks = np.array(range(-1,kernel_size*int(output_neurons**0.5) + 1,kernel_size))
ticks[0] = 0
ax1.set_xticks(ticks); ax2.set_xticks(ticks)
ax1.set_yticks(ticks); ax2.set_yticks(ticks)
ax1.grid(); ax2.grid()
ax1.set_title("RGB receptive fields")
ax2.set_title("LMS receptive fields")

plt.figure()
make_mosaics(lms_weights, all_c)

if False:
    all_c = contour_coord_gauss(lms_weights)
    center_colors = return_center_colors(lms_weights, all_c, remove_mean = False, normalize_sd = False)
    plt.figure()
    plt.imshow(center_colors, cmap = 'gray')
    
    plt.figure()
    plt.plot(center_colors[:,0], color = 'r')
    plt.plot(center_colors[:,1], color = 'g')
    plt.plot(center_colors[:,2], color = 'b')
    plt.title('LMS cones at RF centers', size = 18)
    plt.xlabel('Neurons #', size = 20)
    plt.ylabel('Weight', size = 20)
    plt.axhline(0, color = 'black')

c_near = c_nearest_neighbor(all_c)
lum_bool = on_or_off(lms_weights, all_c)
all_c_on = all_c[:, lum_bool]
all_c_off = all_c[:,np.invert(lum_bool)]
c_near_on = c_nearest_neighbor(all_c_on)
[i for i, n in enumerate(c_near) if n == np.min(c_near_on)]

centers_on = return_center_colors(lms_weights[lum_bool,:,:,:], all_c_on)
centers_off = return_center_colors(lms_weights[np.invert(lum_bool),:,:,:], all_c_off)
centers_on_norm = centers_on/np.expand_dims(np.sum(centers_on**2, axis = 1),1)**0.5
centers_off_norm = centers_off/np.expand_dims(np.sum(centers_off**2, axis = 1),1)**0.5
#centers_norm = centers
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(centers_on_norm[:,0], centers_on_norm[:,1], centers_on_norm[:,2], 'o', color = 'yellow')
ax.plot3D(centers_off_norm[:,0], centers_off_norm[:,1], centers_off_norm[:,2], 'o', color = 'black')
ax.set_xlabel('L'); ax.set_ylabel('M'); ax.set_zlabel('S')
ax.set_title("LMS centers in 3D space after L2 norm")