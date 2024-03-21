# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 14:51:23 2024

@author: David
"""
import torch 
import numpy as np 
from data import KyotoNaturalImages

dataset = KyotoNaturalImages('kyoto_natim', 18, True, "cpu", 1)
dataset_covariance = dataset.covariance().to("cpu")
G = torch.tensor(np.ones(18*18), device = "cpu")

def matrix_spatiotemporal(input: torch.Tensor, W: torch.Tensor, record_C = False):
    # compute C_rx in VAE note page 8.
    # input.shape = [LD, LD], gain.shape = [1 or B, T or 1, J]
    assert input.ndim == 2 and input.shape[0] == input.shape[1]
    L = input.shape[0] // 324
    D = 324
    J = 100
    #L = 3, D = 324, J = 100
    
    #David: I permuted x multiple once so the tensor multiplication 
    #(y = input @ self.w from spatiotemporal) would have consistent dimensions. 
    
    x = input.reshape(L * D, 1, L*D)         # shape = [LD, L, D]
    x = spatiotemporal(x,W)             # shape = [D, 1, J] or [LD, T, J]
    x = x.permute(1, 2, 0)                 # shape = [1, J, D] or [T, J, LD]
    x = x.reshape(-1, L, D)                # shape = [J, 1, D] or [TJ, L, D]
    output_dim = x.shape[0]
    print(x.shape, "hello")
    if D > 1: #edit 1/3/2024
        x = x.reshape(J, 1, L*D) #I did so here. IMPORTANT: Need to put this line for more than 1 channel :)
    x = spatiotemporal(x, W)             # shape = [J, 1, J] or [TJ, T, J]
    x = x.flatten(start_dim=1)             # shape = [J, J]    or [TJ, TJ]
    #if cov == False:
    #x = x.reshape(J, J) #David: and here
    #David: gain.shape = [3,100,100]
    #G = gain.reshape(-1, output_dim)       # shape = [1 or B, J] or [1 or B, TJ]
    #David: G.shape = [100,100]
    #x = G[:, :, None] * x * G[:, None, :]  # shape = [1 or B, J, J] or [1 or B, TJ, TJ]
    #David: x.shape = [100,100,100]
    return x  # this is C_rx


def matrix_spatiotemporal2(input: torch.Tensor, W:torch.Tensor, record_C = False):
    # compute C_rx in VAE note page 8.
    # input.shape = [LD, LD], gain.shape = [1 or B, T or 1, J]
    assert input.ndim == 2 and input.shape[0] == input.shape[1]
    L = input.shape[0] // 324
    D = 324
    J = 100
    #L = 3, D = 324, J = 100
    
    #David: I permuted x multiple once so the tensor multiplication 
    #(y = input @ self.w from spatiotemporal) would have consistent dimensions. 

    x = spatiotemporal(input, W)             # shape = [D, 1, J] or [LD, T, J]
    x = x.permute(1, 0)                 # shape = [1, J, D] or [T, J, LD]
    output_dim = x.shape[0]
    print(x.shape, "hello2")
    x = spatiotemporal(x,W)             # shape = [J, 1, J] or [TJ, T, J]
    #David: gain.shape = [3,100,100]
    #G = gain.reshape(-1, output_dim)       # shape = [1 or B, J] or [1 or B, TJ]
    #David: G.shape = [100,100]
    #x = G[:, :, None] * x * G[:, None, :]  # shape = [1 or B, J, J] or [1 or B, TJ, TJ]
    #David: x.shape = [100,100,100]
    return x  # this is C_rx


def spatiotemporal(input: torch.Tensor, W: torch.Tensor):
    y = input @ W
    return y