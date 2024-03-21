# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 10:31:26 2024
@author: Rache
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torchvision
from data import estimated_covariance
import matplotlib

length = 19
x, y = np.meshgrid(np.array(range(length)), np.array(range(length)))

x1, y1 = x.flatten(), y.flatten()
x2, y2 = np.copy(x1), np.copy(y1)
B = 1
cov = np.zeros([length**2, length**2])
distances = np.zeros([length**2, length**2])
for pos1 in range(x1.shape[0]):
    for pos2 in range(x2.shape[0]):
        distance = np.sqrt((x1[pos1]-x2[pos2])**2 + (y1[pos1]-y2[pos2])**2)
        cov[pos1, pos2] = math.exp(-distance/B)
        distances[pos1, pos2] = distance


############################
sd = 1

dist_x,dist_y = np.meshgrid(np.array(range(5)), np.array(range(5)))
dist_x = dist_x.astype(np.float) - np.mean(dist_x)
dist_y = dist_y.astype(np.float) - np.mean(dist_y)

dist = np.sqrt(dist_x**2 + dist_y**2)
exp_kernel = torch.exp(torch.tensor(-dist/sd, dtype = torch.float)).unsqueeze(0).unsqueeze(0)

noise_uncorr = torch.randn([10000,length,length]).unsqueeze(1)
noise_corr = torch.nn.functional.conv2d(noise_uncorr,exp_kernel, padding = 'same')
#noise_corr = torchvision.transforms.functional.gaussian_blur(noise_uncorr, [5,5], 1)

noise_corr = noise_corr.swapaxes(0,3)
flat_corr = noise_corr.flatten(end_dim=2).numpy()

#flat_corr = noise_corr.flatten(start_dim=1).numpy()
#flat_uncorr = noise_uncorr.flatten(start_dim=1).numpy()



cov2 = np.cov(flat_corr)
matplotlib.use("QtAgg")
plt.imshow(cov)


#B = np.ones([length**2,length**2])*0.01
#roll = np.repeat(np.array(range(length))[np.newaxis,:], repeats = length**2, axis = 0)



#def gauss_dist(x1,y1,roll, B):
#    x2 = np.roll(x1,roll)
#    y2 = np.roll(y1,roll)
#    distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
#    return math.exp(-distance/B)


#result = list(map(gauss_dist,x1,y1,roll, B))







