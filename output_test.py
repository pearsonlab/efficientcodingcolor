# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 16:43:51 2023

@author: Rache
"""
from data import KyotoNaturalImages
from util import cycle
from torch.utils.data import DataLoader
from MosaicAnalysis import Analysis
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from math import atan2

save = '230625-235533'
path = "saves/" + save + "/"

batch_size = 100
n_colors = 3

images = KyotoNaturalImages('kyoto_natim', 18, True, 'cuda', 3)
cov = images.covariance()
hey1 = next(cycle(DataLoader(images, batch_size))).to('cuda')
hey = hey1.reshape([batch_size,n_colors,324])

test = Analysis(path)
test.model.encoder.data_covariance = cov

z, r, C_z, C_zx = test.model.encoder(image = hey)
r = r.cpu().detach().numpy()

hey2 = hey.detach().cpu().numpy()
hey3 = np.swapaxes(hey2, 1, 2)
hey4 = np.concatenate(hey3)
hey5 = np.swapaxes(hey4, 0, 1)
#X = np.reshape(np.concatenate(hey, 1), [3,1800*18])
#Xt = np.swapaxes(X, 0, 1)
cov = np.cov(hey5)
cov_inv = np.linalg.inv(cov)
# ellipse = np.matmul(np.matmul(Xt, cov_inv), X)

#Code by Pranjal:
def draw_ellipse(mu, sig, sig_ell=1, alpha_ell=.1, ax = None):
    size = 22
    u1,s,v1 = np.linalg.svd(sig)
    #angle1 = atan2(v[0,1],v[0,0])*360 / (2*np.pi)
    if ax is not None:
        fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
        ax = fig.add_subplot(111, projection='3d')

    coefs = (np.sqrt(s[0])*sig_ell, np.sqrt(s[1])*sig_ell, np.sqrt(s[2]*sig_ell))  # Coefficients in a0/c x**2 + a1/c y**2 + a2/c z**2 = 1 
    # Radii corresponding to the coefficients:
    rx, ry, rz = 1/np.sqrt(coefs)
    
    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    
    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):
    x = rx * np.outer(np.cos(u), np.sin(v))
    y = ry * np.outer(np.sin(u), np.sin(v))
    z = rz * np.outer(np.ones_like(u), np.cos(v))
    
    # Plot:
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b')
    
    # Adjustment of the axes, so that they all have the same span:
    max_radius = max(rx, ry, rz)
    for axis in 'xyz':
        getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))
    ax.set_xlabel('L', size = size); ax.set_ylabel('M', size = size); ax.set_zlabel('S', size = size)
    plt.show()
        # breakpoint()
        #el = Ellipse((mu[0],mu[1]), width, height, angle)
        #el.set_alpha(alpha_ell)
        #el.set_clip_box(ax.bbox)
        #el.set_facecolor('#304392')  ##ed6713')
        #ax.add_artist(el)
        
    return ax