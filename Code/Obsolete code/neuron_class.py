# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 13:29:22 2023

@author: Rache
"""

class neuron:
    def __init__(self, 
                 weights,
                 id_neuron):
        self.weights = weights[:,:,:,id_neuron]
        pca_inv = get_pca_comps(inv = True)
        self.lms_w = np.tensordot(pca_inv, self.weights, axes = 1)
    def (self):
        1+1 == 2
        