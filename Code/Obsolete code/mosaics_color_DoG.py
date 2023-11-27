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
matplotlib.use('QtAgg')

#Model you want to do the mosaic from
#save = '230502-125114'
#save = '230508-014147'
#save = '230517-143852'
#save = '230519-174509'
#save = '230523-205953'
#save = '230524-182851'
#save = '230530-025727'
#save = '230601-152119'
save = '230602-134703'
save = '230608-155008'
save = '230614-114552'
save = '230616-151057'
save = '230618-015647'
save = '230618-023906'
save = '230619-020142' #This debugged version also has encoder.W direclty embedded 
save = '230620-014248'
path = "saves/" + save + "/"
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




#Import learned model and its weights. Figure out n_colors, kernel_size and output_neurons from weights size. 
cp = torch.load(path + find_last_cp(path)) #bug here plz fix
kernel_centers = cp['model_state_dict']['encoder.kernel_centers'].cpu()
def make_mosaic_DoG(all_c, all_params = None, is_on = None):
    ax = plt.gca()
    ax.set_aspect('equal')
    if all_params is not None:
        n_colors = int(all_params.shape[0] // 4)
    else:
        n_colors = 1
    fig, axes = plt.subplots(1,n_colors)
    if n_colors == 1:
        axes = [axes]
    color = -1
    if n_colors == 3:
        lms_str = ['L cones', 'M cones', 'S cones']
    else:
        lms_str = ['RF centers']
    
    for ax in axes:
        color = color + 1
        color_step = color*4
        for n in range(all_c.shape[0]):
            if is_on[n]:
                marker = 'x'
            else:
                marker = 'o'
            x = all_c[n,0]
            y = all_c[n,1]
            ax.plot(x, y, marker = marker, markersize = 6, color = 'black')
            ax.set_xlim([0,18])
            ax.set_ylim([0,18])
            ax.set_aspect('equal')
            ax.set_title('Mosaic of ' + lms_str[color], size = 12)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            if all_params is not None:
    
                a_pre = all_params[0 + color_step,n]
                b_pre = all_params[1 + color_step,n]
                b = np.sqrt(1/(2*np.exp(b_pre)))
                a = np.sqrt(1/(2*(np.exp(a_pre) + b)))
            
                
                circle_center = plt.Circle((x,y), radius = a, facecolor = 'none', edgecolor = 'r')
                circle_surround = plt.Circle((x,y), radius = b, facecolor = 'none', edgecolor = 'b')
                ax.add_patch(circle_center)
                ax.add_patch(circle_surround)
                
                
                
                
    



all_params = cp['model_state_dict']['encoder.shape_function.shape_params'].cpu()

if False:
    all_d = all_params[5:8,:]
    plt.plot(all_d[0,:], 'r')
    plt.plot(all_d[1,:], 'g')
    plt.plot(all_d[2,:], 'b')

    plt.figure()
    all_c = all_params[2:5,:]
    plt.plot(all_c[0,:], 'r')
    plt.plot(all_c[1,:], 'g')
    plt.plot(all_c[2,:], 'b')
    
    rd = torch.tensor((np.array(range(18)) - 9)**2)
    rd = torch.unsqueeze(rd,1)
    dog = shapes.DifferenceOfGaussianShapeColor(18, 1, all_params[:,0]).shape_function(rd)
    #dog = shapes.DifferenceOfGaussianShape(100, 1, [-3, -0.9, -6]).shape_function(rd)
    
    
    dog = dog.cpu().detach().numpy()
    dog = dog.reshape(18,3)[:,0]
    plt.figure()
    plt.plot(dog)
