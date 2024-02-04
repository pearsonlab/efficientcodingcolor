# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:52:55 2024

@author: David
"""

import torch
from torch import nn
from shapes import get_shape_module, Shape
from MosaicAnalysis import test
import numpy as np
import matplotlib.pyplot as plt

device = "cuda:0"

kernel_centers = test.kernel_centers
kernel_centers = nn.Parameter(torch.tensor(test.kernel_centers, device = "cuda:0"))
#kernel_centers = nn.Parameter(torch.tensor(np.random.normal(8,1,test.kernel_centers.shape), device = "cuda:0"))
#RFs_DoG = test.model.encoder.shape_function(kernel_centers)
DoG_mod = get_shape_module("difference-of-gaussian")(torch.tensor(18, device = device), 3, torch.tensor(300, device = device)).to(device)
params = DoG_mod.shape_params

RFs = torch.tensor(test.w_flat, device = "cuda:0")

optimizer = torch.optim.SGD([kernel_centers, DoG_mod.shape_params], lr = 0.1)

for i in range(100000):
    optimizer.zero_grad()
    RFs_DoG = DoG_mod(kernel_centers)
    loss = torch.sum((RFs - RFs_DoG)**2)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if i%1000 == 0:
        print(loss, i)
    if i == 10000:
        for g in optimizer.param_groups:
            g['lr'] = g['lr']/2
   
    
params_cpu = params.detach().cpu().numpy()
RFs_fit = np.swapaxes(np.reshape(RFs_DoG.detach().cpu().numpy(), [3,18,18,300]), 0, 3)
RFs_og = np.swapaxes(np.reshape(RFs.detach().cpu().numpy(), [3,18,18,300]), 0, 3)

r_coefs = []
for i in range(300):
  fit_flat = RFs_fit[i,:,:,:].flatten()
  og_flat = RFs_og[i,:,:,:].flatten()
  coef = np.corrcoef(og_flat, fit_flat)
  r_coefs.append(coef[1,0])  
  
plt.hist(r_coefs, bins = 100)
