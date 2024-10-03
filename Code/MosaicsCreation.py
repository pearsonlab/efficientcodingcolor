#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 18:22:00 2024

@author: david
"""
import os
import torch
from torch import nn
import numpy as np
from data import KyotoNaturalImages
from model import RetinaVAE, OutputTerms, OutputMetrics
from util import hexagonal_grid
from torch.utils.data import DataLoader
from util import cycle
from datetime import datetime
from tempfile import gettempdir


run_code = False
neurons_per_mosaic = np.array([150,150])
params = np.array([[1,0.3,0.3, 0.5,    1,0.3,0.3,-0.5],
                   [1,0.3,0.3, -0.5,    1,0.3,0.3,0.5]]).T
input_noise = 0.4
output_noise = 3
gain = 1
bias = 5

kernel_size = 18



class Mosaics():
    def __init__(self, params, neurons_per_mosaic, kernel_size, gain, bias, input_noise, output_noise, images = None):
        
        self.batch_size = 128
        self.kernel_size = kernel_size
        self.input_noise = input_noise
        self.output_noise = output_noise
        n_neurons = np.sum(neurons_per_mosaic)
        self.n_neurons = n_neurons
        n_mosaics = neurons_per_mosaic.shape[0]
        self.n_mosaics = n_mosaics
        
        if params.shape[0]%4 == 0:
            n_colors = int(params.shape[0]/4)
        else:
            raise Exception("Number of shape params need to be a multiple of 4")
        self.n_colors = n_colors
        
        circle_masking = True

        self.device = "cuda:0"
        image_restriction = 'True'
        norm_image = False
        
        #This creates an array representing the type for each single neuron. 
        neurons_type = []
        t = 0
        for m in neurons_per_mosaic:
            for n in range(m):
                neurons_type.append(t)
            t += 1
        
        def params_mosaics(params, neurons_per_mosaic):
            params_neurons = np.zeros([params.shape[0], n_neurons])
            if params.shape[1] != n_mosaics:
                raise Exception("params does not have the right shape")
            for n in range(n_neurons):
                t = neurons_type[n]
                params_neurons[:,n] = params[:,t]
            return nn.Parameter(torch.tensor(params_neurons, device = self.device), requires_grad = False)
        
        
        if images is None:
            self.images = KyotoNaturalImages('kyoto_natim', kernel_size, circle_masking, self.device, n_colors, normalize_color = True, restriction = image_restriction, remove_mean = norm_image)
        else:
            self.images = images
        if not hasattr(self.images, 'cov'):
            self.images.covariance()
        
        load = next(cycle(DataLoader(self.images, 100000))).to(self.device)[0:self.batch_size,:,:,:]
        X = load.view(self.batch_size, -1, kernel_size*kernel_size)
        
        model_args = dict(
            kernel_size= kernel_size,
            neurons = n_neurons,
            input_noise = input_noise,
            output_noise = output_noise,
            nonlinearity = "softplus",
            shape = 'difference-of-gaussian',
            individual_shapes = True,
            data_covariance=self.images.cov,
            beta=-0.5,
            rho=1,
            fix_centers = True,
            n_colors = n_colors,
            n_mosaics = n_mosaics,
            corr_noise_sd = 0
        )
        
        self.model_args = model_args
        self.model = RetinaVAE(**model_args).to(self.device)
        self.model.encoder(X, 'Lagrange', 0)
        self.model.encoder.shape_function.shape_params = params_mosaics(params, neurons_per_mosaic)
        self.model.encoder.logA = nn.Parameter(torch.log(torch.tensor(gain, device = self.device)).repeat(n_neurons))
        self.model.encoder.logB = nn.Parameter(torch.log(torch.tensor(bias, device = self.device)).repeat(n_neurons))
        self.model.encoder(X, 'Lagrange', 0)
        
        
    def loss(self, n_cycles):
        FR = []
        MI = []
        for this_cycle in range(n_cycles):
            load = next(cycle(DataLoader(self.images, 100000))).to(self.device)[0:self.batch_size,:,:,:]
            X = load.view(self.batch_size, -1, self.kernel_size*self.kernel_size)
            output: OutputTerms = self.model(X, 'Lagrange', 0) #To update the weights
            metrics: OutputMetrics = output.calculate_metrics(0, 'Lagrange')
            this_MI = metrics.final_loss('Lagrange')
            this_FR = torch.sum(metrics.return_h()**2)
            MI.append(this_MI.item())
            FR.append(this_FR.item())
        self.MI = MI
        self.FR = FR
        return np.mean(MI), np.mean(FR)
    
    def save(self, name):
        train_args = dict(logdir = name,
                        iterations = 1_000,
                        batch_size = self.batch_size,
                        data = "imagenet",
                        kernel_size = self.kernel_size,
                        circle_masking = True,
                        neurons = self.n_neurons,  # number of neurons, J
                        jittering_start = 0, #originally 200000
                        jittering_stop = 0, #originally 500000
                        jittering_interval = 5000,
                        jittering_power = 0.25,
                        centering_weight = 0.02,
                        centering_start = 0, #originally 200000
                        centering_stop = 0, #originally 500000
                        input_noise = self.input_noise,
                        output_noise = self.output_noise,
                        nonlinearity = "softplus",
                        beta = -0.5,
                        n_colors = self.n_colors,
                        shape = 'difference-of-gaussian', # "difference-of-gaussian" for Oneshape case #BUG: Can't use color 1 with "difference-of-gaussian"
                        individual_shapes = True,  # individual size of the RFs can be different for the Oneshape case
                        optimizer = "sgd",  # can be "adam"
                        learning_rate = 0.01, #Consider having a high learning rate at first then lower it. Pytorch has packages for this 
                        rho = 1,
                        maxgradnorm = 20.0,
                        load_checkpoint = None,  # checkpoint file to resume training from
                        fix_centers = True,  # used if we want to fix the kernel_centers to learn params
                        n_mosaics = self.n_mosaics, #Number of mosaics. Only relevant is fix_centers = True
                        whiten_pca_ratio = None, #Default is None
                        device = self.device,
                        firing_restriction = "Lagrange",
                        corr_noise_sd = 0, #Default is 0. Correlated input noise across space 
                        image_restriction = "True", #Default is "True" 
                        flip_odds = 0, #Flips L and S channels with a certain probability. Only works with 2 colors
                        norm_image = False) #Removes the mean from each small image vb  )
        path = "../../saves/artificial_mosaics/" + name
        if not os.path.exists(path):
            os.mkdir(path)
        cp_save = path + "/checkpoint-1000.pt"
        optimizer = train_args['optimizer']
        optimizer_class = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}[optimizer]
        optimizer_kwargs_MI = dict(lr=train_args['learning_rate'])
        optimizer_MI = optimizer_class(self.model.parameters(), **optimizer_kwargs_MI)
        
        torch.save(dict(
                    iteration=0,
                    args=train_args,
                    model_args=self.model_args,
                    model_state_dict=self.model.state_dict(),
                    optimizer_state_dict=optimizer_MI.state_dict(),
                    weights= self.model.encoder.W,
                    MI_matrices = None,
                ), cp_save)
        torch.save(self.model, path + "/model-1000.pt")

if run_code:
    hey = Mosaics(params, neurons_per_mosaic, kernel_size, gain, bias, input_noise, output_noise)
