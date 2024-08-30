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
import random
from util import flip_images

save = '240301-055438_test8'
path = "../../saves/" + save + "/" 
#test = Analysis(path)
#test.get_images()

def get_samples(kernel_size, n_colors = 3):
    img_full = KyotoNaturalImages('kyoto_natim', kernel_size, circle_masking = True, device = 'cpu',n_colors =  n_colors, normalize_color = True, restriction = 'True', remove_mean = False)
    load = next(cycle(DataLoader(img_full, 100000))).to('cpu')
    return load
load = get_samples(100)

def make_psd_1D(psd_2D, plot):
    fig, ax = plt.subplots(1)
    lms = ['L', 'M', 'S']
    rgb = ['r', 'g', 'b']
    jitter = [0,0.003,0]
    size = psd_2D.shape[1]
    lines = []
    for c in range(psd_2D.shape[0]):
        freqs_norm = []
        power = []
        freqs = np.fft.fftfreq(size)
        for i in range(int(size/2)):
            for j in range(int(size/2)):
                freqs_norm.append(np.sqrt(freqs[i]**2 + freqs[j]**2))
                power.append(psd_2D[c, i, j])
        log_freqs = np.log10(freqs_norm)
        log_power = np.log10(power)
        line, = ax.plot(log_freqs + jitter[c], log_power, label = lms[c], color = rgb[c], alpha = 0.5)
        lines.append(line)
    ax.set_xlabel("log10(Frequency norm)", size = 30)
    ax.set_ylabel("log10(power)", size = 30)
    ax.legend(handles = lines, fontsize = 50)
    return log_freqs, log_power 
#f = np.fft.fft2(load)
#f1 = np.fft.fft2(load[:,0,:,:] - load[:,1,:,:])
#f2 = np.fft.fft2(load[:,2,:,:] - (load[:,0,:,:] + load[:,1,:,:])/2)
f1 = np.fft.fft2(load[:,0,:,:]) 
f2 = np.fft.fft2(load[:,1,:,:]) 
f3 = np.fft.fft2(load[:,2,:,:]) 
f = np.stack([f1,f2,f3], axis = 1)
psd_pre = abs(f)**2
psd_2D = np.mean(psd_pre, axis = 0)
make_psd_1D(psd_2D, True)



#load = flip_images(load,0.5, 'cpu')
plt.figure()

var_img = torch.var(load.flatten(1,3), axis = 1)
var_img_L = torch.var(load[:,0,:,:].flatten(1,2),axis=1)
var_img_S = torch.var(load[:,1,:,:].flatten(1,2),axis=1)

mean_img = torch.mean(load.flatten(1,3),axis=1).numpy()
mean_img_L = torch.mean(load[:,0,:,:].flatten(1,2), axis = 1).numpy()
mean_img_S = torch.mean(load[:,1,:,:].flatten(1,2), axis = 1).numpy()
mean_img_L_S = torch.mean((load[:,0,:,:] - load[:,1,:,:]).flatten(1,2)**2, axis = 1).numpy()
#torch.mean(load[mean_img > 1,1,:,:].flatten(1,2)) #Test experiment for mean
#np.logical_or is a function you can use here

load_sv = load[mean_img_L < mean_img_S + 0.4,:,:,:]
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

#COOL STUFF
#condition = np.logical_and(mean_img_L < mean_img_S + 0.5, mean_img_L + 0.5 > mean_img_S).astype(int)
condition = (mean_img_L < mean_img_S + 0.6).astype(int)
plt.figure()
plt.axes().set_aspect('equal')
plt.scatter(mean_img_L, mean_img_S)
plt.plot(np.linspace(-2,2,100), np.linspace(-2,2,100), 'black')
plt.xlabel("Average L inputs (per 18x18 image)", fontsize = 30)
plt.ylabel("Average S inputs (per 18x18 image)", fontsize = 30)
plt.title("Correlation between average L and S inputs from 6200 18x18 images", fontsize = 30)
#plt.scatter(mean_img_L, mean_img_S, c = np.array(var_img < 1).astype(int))

n_samples = 10000
samples = random.sample(range(1, load[:,0,:,:].flatten().shape[0]), n_samples)

L_pixels = load[:,0,:,:].flatten()[samples]
S_pixels = load[:,1,:,:].flatten()[samples]
plt.figure()
plt.axes().set_aspect('equal')
plt.scatter(L_pixels, S_pixels)#, c = condition)
plt.plot(np.linspace(-2,2,100), np.linspace(-2,2,100), 'black')
plt.xlabel("Pixel L inputs", fontsize = 30)
plt.ylabel("Pixel S inputs", fontsize = 30)
plt.title("Correlation between L and S inputs for " + str(n_samples) + " pixels", fontsize = 30)
plt.figure()
plt.hist(L_pixels - S_pixels, bins = 200)
plt.xlabel("Sample of L - S pixels", fontsize = 30)
plt.ylabel("Number of pixels", fontsize = 30)
plt.axvline(0, color = 'black')
#plt.scatter(mean_img_L, mean_img_S, c = np.array(var_img < 1).astype(int))