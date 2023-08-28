# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 15:30:37 2022

@author: David
"""
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import torch
from util import cycle, kernel_images
from PIL import Image
matplotlib.use('QtAgg')

#Model you want to do the mosaic from
#save = '220907-223053'
save = '220925-235656'

#Import learned model and its weights
cp = torch.load("saves/" + save +  "/checkpoint-1000000.pt")
kernel_size = cp['args']['kernel_size']
weights = cp['model_state_dict']['encoder.W'].cpu().detach().numpy()

#RFs = kernel_images(weights, kernel_size)

#Separates list of neurons into ON and OFF neurons, based on their strongest (absolute) weight value
def select_on_off(W):
    neurons_types = abs(np.min(W, axis = 0)) < np.max(W, axis = 0)
    on_neurons = W[:,neurons_types]
    off_neurons = W[:,np.invert(neurons_types)]
    return on_neurons, off_neurons
    
#Does a contour plot for each neuron and return its contour line coordinates
def contour_coord(W, pathway, levels = [0.25,0.3]):
    on_neurons, off_neurons = select_on_off(W)
    if pathway == 'on':
        neurons = on_neurons
    elif pathway == 'off':
        neurons = abs(off_neurons)
    else:
        raise Exception
    neurons = np.reshape(neurons, newshape = [18,18,neurons.shape[1]])
    n_neurons = neurons.shape[2]
    all_c = np.zeros([2, n_neurons])
    for n in range(n_neurons):
        fig = plt.contour(neurons[:,:,n], levels = levels)
        plt.close()
        c = center_of_mass(fig.allsegs[-1][0])
        all_c[:,n] = [c[0], c[1]]
        
    return all_c

#Finds the center of mass of a neuron based on the coordinates of its contour plot
def center_of_mass(X):
    x_mean = np.mean(X[:,0])
    y_mean = np.mean(X[:,1])
    return [x_mean,y_mean]

#Makes mosaic from center of mass coordinates
def make_mosaic(all_c, marker):
    markers = ['o', 'x']
    for pathway in ['on', 'off']:
        c = contour_coord(weights, pathway)
        
        ax = plt.gca()
        for n in range(all_c.shape[1]):
            ax.plot(c[0,n], c[1,n], marker = marker, markersize = 6, color = 'black')

def make_mosaic(weights):
    c_on = contour_coord(weights, pathway = 'on')
    c_off = contour_coord(weights, pathway = 'off')
    show_mosaic(c_on, 'o')
    plt.savefig('on_mosaic.png'); plt.close()
    show_mosaic(c_off, 'x')
    plt.savefig('off_mosaic.png'); plt.close()
    

