# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 13:13:03 2023

@author: Rache
"""
import numpy as np
from sklearn.mixture import GaussianMixture

n_tries = 50
best_aic = 0
gauss = GaussianMixture(n_components = 6, n_init = 50).fit(np.swapaxes(test.pca_transform, 0,1))
aic = gauss.aic(np.swapaxes(test.pca_transform,0,1))


ax = test.plot3D(np.transpose(test.pca_transform), color_type = False)
ax.plot(test.gauss_pca.means_[:,0], test.gauss_pca.means_[:,1], test.gauss_pca.means_[:,2], color = 'red')