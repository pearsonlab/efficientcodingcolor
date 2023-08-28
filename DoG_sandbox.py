# -*- coding: utf-8 -*-
"""
Created on Tue May 23 13:09:11 2023

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
import shapes
import mosaics_color_DoG
#matplotlib.use('QtAgg')

#Model you want to do the mosaic from
#save = '230601-003809'
#save = '230601-152119'
save = '230602-134703'
save = '230606-020131'
save = '230606-150041'
save = '230606-230054'
save = '230607-152019'
save = '230608-155008'
save = '230612-032704'
#save = '230614-114552'
#save = '230615-032642'
save = '230616-020802'
save = '230616-151057'

#save = '230617-014438'
save = '230617-185529'
save = '230617-193940'
save = '230618-015647'
save = '230618-023906' #**This should work!!!!!!!!! bug was fixed
save= '230619-013529'
save = '230619-020142' #This debugged version also has encoder.W direclty embedded 
save = '230619-234711'
path = "saves/" + save + "/"

kernel_size = 18
neuron =26
color = 2

#Open log file to figure out if this simulation had PCA pre-processing or not
logs = pd.read_csv('./Simulation_logs.csv')
#makepca_preprocess = logs['PCA'][logs['ID'] == save].values[0] == 'TRUE'

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


def get_all_params_time(path, interval = 1000):
    max_cp = find_last_cp(path)
    max_iteration = int(max_cp[11:-3])
    iterations = np.array(range(interval,max_iteration,interval))
    i = 0; first = True
    for iteration in iterations:
        cp = torch.load(path + 'checkpoint-' + str(iteration) + '.pt')
        all_params = cp['model_state_dict']['encoder.shape_function.shape_params'].cpu().detach().numpy()
        if first:
            all_params_time = np.zeros([iterations.shape[0], all_params.shape[0], all_params.shape[1]])
            first = False
        all_params_time[i,:,:] = all_params
        i = i + 1
    return all_params_time

def get_kernels_time(path, interval = 1000):
    max_cp = find_last_cp(path)
    max_iteration = int(max_cp[11:-3])
    iterations = np.array(range(interval,max_iteration,interval))
    i = 0; first = True
    for iteration in iterations:
        cp = torch.load(path + 'checkpoint-' + str(iteration) + '.pt')
        all_kernels = cp['model_state_dict']['encoder.kernel_centers'].cpu().detach().numpy()
        if first:
            all_kernels_time = np.zeros([iterations.shape[0], all_kernels.shape[0], all_kernels.shape[1]])
            first = False
        all_kernels_time[i,:,:] = all_kernels
        i = i + 1
    return 



def make_mosaic_DoG_time(path, interval = 1000):
    kernels_time = get_kernels_time(path, interval = interval)
    all_params_time = get_all_params_time(path, interval = interval)
    os.mkdir('Figures/' + save)
    for i in range(kernels_time.shape[0]):
        if i%100 == 0:
            print(i)
        plt.figure()
        mosaics_color_DoG.make_mosaic_DoG(kernels_time[i,:,:], marker = 'x', all_params = all_params_time[i,:,:])
        plt.savefig('Figures/' + save + '/' + 'mosaic_' + str(i) + '.png')
        plt.close('all')


#Import learned model and its weights. Figure out n_colors, kernel_size and output_neurons from weights size. 
cp = torch.load(path + find_last_cp(path)) #bug here plz fix
#model_cp = torch.load(path + 'model-100000.pt')
kernel_centers = cp['model_state_dict']['encoder.kernel_centers'].cpu()
all_params = cp['model_state_dict']['encoder.shape_function.shape_params'].cpu()
n_colors = int(all_params.shape[0] / 4)
n_neurons = all_params.shape[1]

half_size = int(kernel_size/2)
x = torch.arange(-half_size,half_size); y = torch.arange(-half_size,half_size)
grid_x, grid_y = torch.meshgrid(x, y)
rd = grid_x.flatten()**2 + grid_y.flatten()**2
rd = torch.unsqueeze(rd,1)

if n_colors == 3:
    #dog = shapes.DifferenceOfGaussianShapeColor(size, 1, [-3, -3, 0.0, 10, 0.0, 1,1,0]).shape_function(rd)
    dog = shapes.DifferenceOfGaussianShape(kernel_size, n_colors, 1, all_params[:,neuron], read = True).shape_function(rd)
else:
    #dog = shapes.DifferenceOfGaussianShape(size, 1, [-3,-3,0.0002]).shape_function(rd)
    dog = shapes.DifferenceOfGaussianShape(kernel_size, n_colors, 1, all_params[:,neuron]).shape_function(rd)
dog = dog.reshape([n_colors, kernel_size,kernel_size])
dog = torch.swapaxes(dog,0,2)
#dog = dog / dog.norm(dim=[0,1], keepdim=True)
dog = (dog - torch.min(dog))/(torch.max(dog) - torch.min(dog))

dog_np = dog.cpu().detach().numpy()
lms_to_rgb = get_matrix(matrix_type = 'lms_to_rgb')

if n_colors == 3:
    dog_rgb = np.tensordot(lms_to_rgb, np.swapaxes(dog_np, 0, 2), axes = 1)
    dog_rgb = np.swapaxes(dog_rgb, 0, 2)
    fig, [ax1, ax2, ax3] = plt.subplots(1,n_colors) 
    ax1.imshow(dog_np[:,:,0])
    ax2.imshow(dog_np[:,:,1])
    ax3.imshow(dog_np[:,:,2])
    plt.figure()
    plt.imshow(dog_rgb)
else:
    plt.imshow(dog_np[:,:,0], 'gray')


plt.figure()
all_params_time = get_all_params_time(path)
plt.plot(all_params_time[:,color,neuron], 'r')
plt.plot(all_params_time[:,color + n_colors,neuron], 'b')
#plt.title('red = a = log(1/center_std), blue = b = log(1/surround_std)', size = 18)
#plt.xlabel('Epoch (x1000)', size = 18)
#plt.ylabel('Input Parameter value', size = 18)



