# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 09:47:09 2023

@author: Rache
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import scipy.optimize as opt
import scipy.signal as ssig
from scipy.interpolate import interp1d
from scipy.special import softmax
import scipy


def softplus(x, beta=1): return np.log1p(np.exp(-np.abs(beta * x)))/beta + np.maximum(x, 0)

plt.rcParams.update({
    "text.usetex": True,
})

sns.set_context('talk')

np.random.seed(12346)

L = 10  # linear size of space in one dimension
C = 3 # number of channels
N = 1001 # number of points
zz = np.linspace(-L/2, L/2, N)
dz = zz[1] - zz[0]

sigin = 0.4
sigout = 1.25
A = 100
alpha = 1.3

def soft_bandpass(lo, hi, freqs, stiffness=10):
    if lo <= 0:
        return scipy.special.expit(stiffness * (hi - freqs))
    else:
        return scipy.special.expit(stiffness * (freqs - lo)) * scipy.special.expit(stiffness * (hi - freqs))

def C(k, o):
    return A/(np.abs(k)**alpha * np.abs(o)**2)

def filter(A, sigin, sigout, nu, k_lims=None, o_lims=None):
    def v_opt(k, omega):
        sqrt_piece = np.sqrt(1 + (4/nu) * (sigin**2/sigout**2) * k**alpha)
        v2 = 0.5 * (sqrt_piece + 1) * A / (A + sigin**2 * k**alpha) - 1
        v2 = np.sqrt(np.maximum(v2, 0) * sigout**2/sigin**2)
        
        if k_lims:
            unit_cell_k = soft_bandpass(k_lims[0], k_lims[1], k)
            v2 *= unit_cell_k
        if o_lims:
            unit_cell_o = soft_bandpass(o_lims[0], o_lims[1], omega)
            v2 *= unit_cell_o
            
        return v2
    return v_opt