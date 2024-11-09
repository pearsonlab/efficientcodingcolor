#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 13:32:00 2024

@author: david
"""


"""
Notes August 29th:
The PSD is the diagonal of Cx across colors
The next step is to get Cx for different channels (e.g. L vs S), which requires multiplying one channel with another in fourier space (instead of taking the square)
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
n_channels = 3 # number of channels
N = 1001 # number of points
n_lags = 10
zz = np.linspace(-L/2, L/2, N)
dz = zz[1] - zz[0]

sigin = 0.4
sigout = 1.25
A = np.array([3,2,1])
#A = 100

alpha = 1.3

def soft_bandpass(lo, hi, freqs, stiffness=10):
    if lo <= 0:
        return scipy.special.expit(stiffness * (hi - freqs))
    else:
        return scipy.special.expit(stiffness * (freqs - lo)) * scipy.special.expit(stiffness * (hi - freqs))

#Create Cx assuming the structure of spatiotemporal PSD
def C(k, w, c):
    output = A/(np.abs(k)**alpha * w**2) 
    if c == 0 or c == 1:
        return output + 1
    elif c == 2 and k > 5 and w < 2:
        return output + 2
    else:
        return output

#This is the filter I need. Equivalent to Formula 34 in John's notes if we replace C by lambda
def filter(lambdas, nu, k_lims=None):
    def v_opt(k, o):
        lam = np.minimum(lambdas(k,o), 1e32)
        sqrt_piece = np.sqrt(lam**2 + (4/nu) * (sigin**2/sigout**2) * lam)
        v2 = 0.5 * (sqrt_piece + lam) / (sigin**2 + lam) - 1
        v2 = np.sqrt(np.maximum(v2, 0) * sigout**2/sigin**2)
        
        if k_lims:
            unit_cell_k = soft_bandpass(k_lims[0], k_lims[1], k)
            v2 *= unit_cell_k
        
        return v2 #- np.min(v2)
    return v_opt

eigvals = np.zeros([L, n_lags, n_channels])
eigvects = np.zeros([L, n_lags, n_channels])
#Solve generalized eigenvalue problem of 
#IMPORTANT: These are the eigenvalues/eigenvectors of Cx + Cin. Need to add input noise here!!
for space_freq in range(L):
        for temp_freq in range(n_lags):
            Cx_color = Cx_fft[space_freq,temp_freq,:,:]
            eig = scipy.linalg.eigh(Cx_color)
            eigvals[space_freq,temp_freq,:] = eig[0]
            eigvects[space_freq, temp_freq, :,:] = eig[1]


#This is formula 10 from the NeurIPS paper. Computes the optimal solution, but requires knowing nu
def filter2(A_all, sigin, sigout, nu, k_lims=None, o_lims=None):
    def v_opt(k, channel):
        A = A_all[channel]
        sqrt_piece = np.sqrt(1 + (4/nu) * (sigin**2/sigout**2) * k**alpha/A)
        v2 = 0.5 * (sqrt_piece + 1) * A / (A + sigin**2 * k**alpha) - 1
        v2 = np.sqrt(np.maximum(v2, 0) * sigout**2/sigin**2)
        if k_lims:
            unit_cell_k = soft_bandpass(k_lims[0], k_lims[1], k)
            v2 *= unit_cell_k
        return v2
    return v_opt

def make_k(J):
    """k is J x 2, with J the number of neurons/frequencies"""
    M = int(np.sqrt(J))  # find biggest square array that fits inside
    m1, m2 = np.meshgrid(range(M), range(M))
    mm = np.stack([m1.ravel(), m2.ravel()], axis=1)
    
    return (2 * np.pi/L) * mm

#Formula 30 in NeurIPS supplementary
def filter_power_discrete(nu, J):
    k = make_k(J)
    
    ktilde = np.linalg.norm(k, axis=1)**alpha * sigin**2

    Pj = np.maximum(0, sigout**2 /(2 * (ktilde + 1e-16)) * (np.sqrt(1 + 4 * ktilde/(nu * sigout**2)) - 1) - sigout**2)

    return np.sum(Pj)

#The power constraint to find nu. 
def excess_power(lognu, J, P):
    return np.log(P / np.max([filter_power_discrete(np.exp(lognu), J), 1e-32]))

#Find optimal nu from formula 30 so we can plug it into the filter function (formula 10). 
def optimal_lognu(J, P):
    power_constraint = {'type': 'ineq', 'fun': excess_power, 'args': (J, P)}
    res = opt.minimize(lambda lognu: lognu,1, #bounds=[(1e-16, np.inf)], 
                       constraints=[power_constraint])
    if res.success:
        return res.x
    else:
        print("Optimizer failed to converge!")
        return None

#Formula 29 in NeurIPS paper supplementary
def information(J, P):
    log_nu = optimal_lognu(J, P)
    nu = np.exp(log_nu)
    
    k = make_k(J)
    
    ktilde = np.linalg.norm(k, axis=1)**alpha * sigin**2 

    numer = np.maximum(0, sigout**2 /(2 * (ktilde + 1e-16)) * (np.sqrt(1 + 4 * ktilde/(nu * sigout**2)) - 1) - sigout**2) + sigout**2
    
    denom = np.maximum(0, sigout**2 /(2 * (ktilde + 1)) * (np.sqrt(1 + 4 * ktilde/(nu * sigout**2)) + 1) - sigout**2) + sigout**2
    
    return np.sum(np.log(numer) - np.log(denom))