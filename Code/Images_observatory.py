# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:23:32 2024

@author: David
"""

from data import KyotoNaturalImages
from torch.utils.data import DataLoader
from util import cycle, kernel_images
from MosaicAnalysis import Analysis
import torch
import numpy as np
import matplotlib.pyplot as plt

save = '240301-055438_test2'
path = "../../saves/" + save + "/" 
#test = Analysis(path)
#test.get_images()

img_full = KyotoNaturalImages('kyoto_natim', 18, False, 'cpu', 2)

load = next(cycle(DataLoader(img_full, 100000))).to('cpu')
plt.figure()

var_img = torch.var(load.flatten(1,3), axis = 1)
#np.logical_or is a function you can use here

load_sv = load[np.logical_and(var_img > 0.9, var_img < 1), :,:,:]
load_pos = load_sv[torch.mean(load_sv.flatten(1,3), axis = 1) > 0,:,:,:]
load_neg = load_sv[torch.mean(load_sv.flatten(1,3), axis = 1) < 0,:,:,:]
print(torch.mean(load_pos[:,0,:,:]), torch.mean(load_pos[:,1,:,:]), load_pos.shape[0])
print(torch.mean(load_neg[:,0,:,:]), torch.mean(load_neg[:,1,:,:]), load_neg.shape[0])
#torch.mean(load[torch.var(load.flatten(1,3), axis = 1) < 0.1, 1,:,:])
#np.corrcoef(torch.mean(load[:,0,:,:].flatten(1,2), axis = 1), torch.mean(load[:,1,:,:].flatten(1,2), axis = 1))


#load_norm = torch.zeros(load.shape)
#for n in range(load.shape[0]):
#    load_norm[n,:,:,:] = load[n,:,:,:] - torch.mean(load[n,:,:,:])

#np.corrcoef(torch.mean(load_norm[:,0,:,:].flatten(1,2), axis = 1), torch.mean(load_norm[:,1,:,:].flatten(1,2), axis = 1))

torch.sum(torch.mean(load.flatten(1,3), axis = 1)**2 + torch.var(load.flatten(1,3), axis = 1))
torch.sum(load**2) /(18*18*2*6200)
