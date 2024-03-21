# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 12:50:08 2024

@author: David
"""
import torch
import os
from data import KyotoNaturalImages
from model import RetinaVAE, OutputTerms, OutputMetrics
from torch.utils.data import DataLoader
from util import cycle, kernel_images
import numpy as np


model = test.model

h_exp = 0
firing_restriction = "Lagrange"
record_C = True



batch_size = 128
dataset = KyotoNaturalImages('kyoto_natim', 18, True, "cuda:0", 1)
data_loader = DataLoader(dataset, batch_size)
data_iterator = cycle(data_loader)

model.encoder.data_covariance = dataset.covariance().to("cuda:0")

batch = next(data_iterator).to("cuda:0")

output: OutputTerms = model(batch, h_exp, firing_restriction, record_C = record_C) 


W = test.model.encoder.W.detach().cpu().numpy()
C = dataset.covariance().detach().cpu().numpy()

WCxW = np.matmul(np.matmul(np.transpose(W), C), W)