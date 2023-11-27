# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 14:03:36 2023

@author: Rache
"""
import numpy as np
import matplotlib.pyplot as plt
from analysis_utils import closest_divisor
import torch
from torch import nn

def make_kernel_centers():
    n_neurons = 500
    n_dist, n_angles = closest_divisor(n_neurons)
    max_length = int(12/2)
    all_dist = np.linspace(0,max_length, n_dist)
    all_angles = np.linspace(0, np.pi*2, n_angles)
    all_kernel_centers = np.zeros([n_dist, n_angles, 2])
    for dist_index in range(n_dist):
        for angle_index in range(n_angles):
            angle = all_angles[angle_index]
            dist = all_dist[dist_index]
            angle_dist = abs(all_angles[1] - all_angles[0])
            if dist_index%2 == 0:
                phase = 0
            else:
                phase = angle_dist
            print(phase)
            x = np.cos(angle + phase/2)*dist
            y = np.sin(angle + phase/2)*dist
            all_kernel_centers[dist_index,angle_index, 0] = x
            all_kernel_centers[dist_index,angle_index, 1] = y
            plt.plot(x,y,'o')
    #return all_kernel_centers

n_neurons = 50
kernel_size = 12
n_epochs = 1
#n_dist, n_angles = closest_divisor(n_neurons)
n_x,n_y = closest_divisor(n_neurons)

size = int(kernel_size/2)
kernel_centers = torch.Tensor(np.random.uniform(-size, size, [n_neurons,2]))

centers_init = torch.stack([kernel_centers[:,0], kernel_centers[:,1]])
centers_init = torch.swapaxes(centers_init,0,1)
#r_init = torch.Tensor(np.linspace(-size,size, n_dist))
#angles_init = torch.Tensor(np.linspace(0, np.pi, n_angles))
x_init = torch.Tensor(np.linspace(-size,size,n_x))
y_init = torch.Tensor(np.linspace(-size,size,n_y))
#r = nn.Parameter(r_init)
#angles = nn.Parameter(angles_init)
x = nn.Parameter(x_init)
y = nn.Parameter(y_init)

optimizer = torch.optim.Adam([x,y], lr=0.0000000001)

for epoch in range(n_epochs):
    for neuron in range(n_neurons):
        x_grid, y_grid = torch.meshgrid(x,y)
        optimizer.zero_grad()
        x_clamp = x_grid.clamp(-8,8)
        x_clamp_flat = x_clamp.flatten(0,1)
        y_clamp = y_grid.clamp(-8,8)
        y_clamp_flat = y_clamp.flatten(0,1)
        kernel_centers = torch.stack((x_clamp_flat,y_clamp_flat),1)
        dist = torch.cdist(kernel_centers, kernel_centers)
        dist_nonzero = dist + torch.diag(torch.Tensor(np.repeat(np.inf,n_neurons)))
        min_dist, min_index = torch.min(dist_nonzero, dim = 1)
    
    #dist = torch.mean(torch.cdist(kernel_centers, kernel_centers))
    
        optimizer.zero_grad()
        loss = 1/min_dist[neuron]
        #loss = torch.mean(-min_dist)
        
        loss.backward()
        optimizer.step()
    if epoch%100 == 0:
        print(torch.mean(min_dist), epoch)
        
for neuron in range(n_neurons):
    centers_cpu = kernel_centers.detach().cpu().numpy()
    plt.plot(centers_cpu[neuron,0], centers_cpu[neuron,1], 'o', color = 'black')
    plt.gca().set_aspect('equal')
