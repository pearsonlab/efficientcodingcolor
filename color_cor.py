# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 14:22:00 2023

@author: David
"""
import numpy as np
import matplotlib
from torch.utils.data import Dataset
from data import KyotoNaturalImages
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
matplotlib.use('QtAgg')

Kyoto = KyotoNaturalImages(root = 'kyoto_natim', kernel_size = 18, circle_masking = True, device = 'cuda', n_colors =3)
images = Kyoto.images
n_images = len(images)
n_colors = 3
colors = [0,1]
color_names = ['L', 'M', 'S']

compute_coefficients = False

all_a = np.linspace(0,0.5,26)


def flatten_images(images):
    n_images = len(images)
    images_reshaped = np.zeros([n_colors, 0])
    for n_color in range(n_colors):
        print(n_color)
        color = np.array([])
        for i in range(n_images):
            color = np.append(color, images[i][n_color,:,:].flatten().cpu())
        if n_color == 0:
            stack_length = color.shape[0]
            images_reshaped = np.zeros([n_colors, stack_length])
        images_reshaped[n_color,:] = color
    return images_reshaped
images_reshaped = flatten_images(images)

#color1 = np.array(images[0][0,50,:].cpu())
#color1_reshape = color1.reshape([color1.shape[0],1])
#color2 = np.array(images[0][1,50,:].cpu())


color_length = images_reshaped.shape[1]
color_subset = np.array(np.random.binomial(1,0.01,color_length), dtype = bool)
color1 = images_reshaped[colors[0],color_subset]
color2 = images_reshaped[colors[1],color_subset]
color3 = images_reshaped[2, color_subset]
subset_length = np.sum(color_subset)


def compute_coefs(color1, color2, all_a):
    lm_coefs = np.array([])
    MI_coefs = np.array([])
    subset_length = len(color1)
    if len(color1) != len(color2):
        raise TypeError("Both colors are not of the same length")
    for a in all_a:
        if abs(a%0.1) < 0.00001:
            print(a)
        color_sum = a*(color1 + color2)
        lm_coef = np.corrcoef(color1 - color_sum, color2 - color_sum)[0][1]
        color1_preshape = color1 - color_sum
        color1_reshape = color1_preshape.reshape([subset_length,1])
        MI_coef = mutual_info_regression(color1_reshape, color2 - color_sum)
        lm_coefs = np.append(lm_coefs,lm_coef)
        MI_coefs = np.append(MI_coefs, MI_coef)
    return lm_coefs, MI_coefs

if compute_coefficients:
    lm_coefs, MI_coefs = compute_coefs(color1, color2, all_a)
    
    MI_min_index = np.where(MI_coefs == min(MI_coefs))[0][0]
    MI_min_a = all_a[MI_min_index]
    corr_0_index = np.where(abs(lm_coefs) == min(abs(lm_coefs)))[0][0]
    corr_0_a = all_a[corr_0_index]
    
    fig, [ax1, ax2] = plt.subplots(1,2)
    ax1.plot(all_a, lm_coefs)
    ax1.axhline(y = 0, linestyle = '--')
    ax1.set_ylabel('Pearson correlation coefficient')
    ax1.set_xlabel('a')
    ax2.plot(all_a, MI_coefs)
    ax2.set_ylabel('Mutual information')
    ax2.set_xlabel('a')
    fig.suptitle("Relationship between " + color_names[colors[0]] + " and " + color_names[colors[1]] + " cone responses. " + "\n" + color_names[colors[0]] + " = " + color_names[colors[0]] + " - a*(" + color_names[0] + " + " + color_names[1] + ")" + "\n" + color_names[colors[1]] + " = " + color_names[colors[1]] + " - a*(" + color_names[0] + " + " + color_names[1] + ")")
    
    ax1.axvline(x = corr_0_a, linestyle = '--')
    ax2.axvline(x = MI_min_a, linestyle = '--')
    ax2.axhline(y = min(MI_coefs), linestyle = '--')
    

all_colors = np.transpose(images_reshaped)
pca = PCA(n_components = 3)
pca.fit(all_colors)
print(pca.components_)