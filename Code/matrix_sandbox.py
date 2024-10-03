# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 14:06:27 2024

@author: David
"""

from analysis_utils import get_samples
import torch
import numpy as np
from MosaicAnalysis import Analysis
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#save2 =  '240614-003858'
save2 = '240301-055438'
analysis = Analysis(save2)
analysis(3,5,4)

n_samples = 128
n_neurons = 300
input_noise = 0.2
output_noise = 2
normalize_color = True
whiten_pca = [50,50]

analysis.get_responses(normalize_color)

if whiten_pca is not None:
    analysis.images.pca_color()
    analysis.images.whiten_pca(whiten_pca)

#Get X: 
#if not 'load' in locals():
load = get_samples(18, 2, normalize_color = normalize_color)[0:n_samples,:,:,:]
x = load.view(n_samples, -1, 18*18)
nx = input_noise * torch.randn_like(x) 
x_noise = x + nx

X = x.flatten(1,2).cuda()
X_noise = x_noise.flatten(1,2).cuda()

#Compute WCinW and etc:
W = analysis.model.encoder.W[:,0:n_neurons]
WT = np.swapaxes(W,0,1)
Cin = torch.diag(torch.tensor(np.repeat(input_noise, W.shape[0]), device = "cuda:0")).float()
Cx = analysis.images.cov
Cxin = Cx + Cin
Cout = torch.diag(torch.tensor(np.repeat(output_noise, W.shape[1]), device = "cuda:0")).float()
WCxW = torch.matmul(WT, torch.matmul(Cx,W))
WCinW = torch.matmul(WT, torch.matmul(Cin.float(), W))
WCxinW = torch.matmul(WT, torch.matmul(Cxin, W))

#Compute the responses before and after the softplus. Point is to get G:
y = torch.matmul(X_noise,W)
nr = output_noise * torch.randn_like(y)
y_nr = y + nr 
gain = torch.tensor(np.repeat(analysis.gain[np.newaxis,0:n_neurons], n_samples, 0), device = 'cuda:0')
bias = torch.tensor(np.repeat(analysis.bias[np.newaxis,0:n_neurons], n_samples, 0), device = "cuda:0")
resp = gain * F.softplus(y_nr - bias, beta = 2.5)
G_pre = gain * F.sigmoid(2.5 * y_nr - bias)
resp = resp.cpu().detach().numpy()

for b in range(G_pre.shape[0]):
    G= torch.diag(G_pre[b,:]) #This takes the average of the responses. **** This is an approximation 
    GWCxWG = torch.matmul(G, torch.matmul(WCxW,G))
    GWCinWG = torch.matmul(G, torch.matmul(WCinW,G))
    GWCxinWG = torch.matmul(G, torch.matmul(WCxinW,G))

    #Compute eigenvals and MI:
    eig = torch.linalg.eig(GWCxinWG + Cout)
    eig_noise = torch.linalg.eig(GWCinWG + Cout)
    eigvals = torch.real(eig[0]).cpu().detach().numpy()
    eigvals_noise = torch.real(eig_noise[0]).cpu().detach().numpy()
    eigvects = torch.real(eig[1]).cpu().detach().numpy()

#Plot eigenvals on a log scale:
plt.plot(np.sort((np.log(eigvals_noise)))[::-1])
plt.xlabel("Index", size = 30)
plt.ylabel("log(Eigenvalue)", size = 30)
plt.title("GW$^T$(C$_{in}$)WG + C$_{out}$", size = 50)
numerator = np.sum(np.log(eigvals))
denominator = np.sum(np.log(eigvals_noise))
print(numerator, denominator, numerator - denominator)

#print(round(np.mean(resp[:,test2.type == 0]), 3), round(np.mean(resp[:, test2.type == 1]), 3),
#      round(np.mean(resp[:,test2.type == 2]), 3), round(np.mean(resp[:, test2.type == 3]), 3))

colors = ['r', 'b', 'g', 'y']
labels = ['OFF', '+S -L', 'ON', '+L -S']

resp_mean = np.mean(resp, axis = 0)
bins = np.linspace(np.min(resp_mean), np.max(resp_mean), 50)
fig, axes = plt.subplots(4)
for t in range(4):
    axes[t].hist(resp_mean[analysis.type == t], bins = bins, alpha = 0.5, color = colors[t], label = labels[t], edgecolor = 'black')
handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in colors]
plt.legend(handles, labels, fontsize = 40, loc = "upper right")
fig.suptitle("Distribution of mean responses per neuron", size = 50)

y_mean = np.mean(y_nr.cpu().detach().numpy(), axis = 0)
bins = np.linspace(np.min(y_mean), np.max(y_mean), 20)
fig, axes = plt.subplots(4)
for t in range(4):
    axes[t].hist(y_mean[analysis.type == t], bins = bins, alpha = 0.5, color = colors[t], label = labels[t], edgecolor = 'black')
handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in colors]
plt.legend(handles, labels, fontsize = 40, loc = "upper right")
fig.suptitle("Input to softplus distribution", size = 50)

fig, axes = plt.subplots(4)
bins = np.linspace(np.min(analysis.gain), np.max(analysis.gain), 50)
for t in range(4):
    axes[t].hist(analysis.gain[analysis.type == t], bins = bins, alpha = 0.5, color = colors[t], label = labels[t], edgecolor = 'black')
handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in colors]
plt.legend(handles, labels, fontsize = 40, loc = "upper right")
fig.suptitle("Gain distribution", size = 50)

fig, axes = plt.subplots(4)
bins = np.linspace(np.min(analysis.bias), np.max(analysis.bias), 20)
for t in range(4):
    axes[t].hist(analysis.bias[analysis.type == t], bins = bins, alpha = 0.5, color = colors[t], label = labels[t], edgecolor = 'black')
handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in colors]
plt.legend(handles, labels, fontsize = 40, loc = "upper right")
fig.suptitle("Bias distribution", size = 50)

