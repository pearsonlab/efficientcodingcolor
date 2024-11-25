# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:27:40 2024

@author: Rache
"""

#This version uses the Cx estimated from natural movies 
import numpy as np
import os
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib 
import pickle


import seaborn as sns
import scipy.optimize as opt
import scipy.signal as ssig
from scipy.interpolate import interp1d
from scipy.special import softmax
from scipy.interpolate import RectBivariateSpline, BivariateSpline
import scipy

def softplus(x, beta=1): return np.log1p(np.exp(-np.abs(beta * x)))/beta + np.maximum(x, 0)

plt.rcParams.update({
    "text.usetex": True,
})

sns.set_context('talk')


np.random.seed(12346)


#sigout = 1.25
sigout = 1
#A = np.array([100])

def soft_bandpass(lo, hi, freqs, stiffness=10):
    if lo <= 0:
        return scipy.special.expit(stiffness * (hi - freqs))
    else:
        return scipy.special.expit(stiffness * (freqs - lo)) * scipy.special.expit(stiffness * (hi - freqs))

#Original version of the code
#def C_prev(k, o):
#    return A/(np.abs(k)**alpha * np.abs(o)**2)

#def C(k,o,i):
#    eigval = A[i]/(np.abs(k)**alpha * np.abs(o)**2)  
#    return eigval

def import_C():
    with open(os.getcwd() + "/../../Cx.pkl", 'rb') as f:
        x = pickle.load(f)
    return x
hey = import_C()
n_time_freqs = hey['Cx_eigvals'].shape[1]//2
n_space_freqs = hey['Cx_eigvals'].shape[0]
Cx_pre = hey['Cx_eigvals'][:,0:n_time_freqs,:]
Cx_pre = np.flip(Cx_pre, axis = 2)
U = hey['Cx_eigvects'][:,0:n_time_freqs,:]
U = np.flip(U, axis = 3)
for i in range(U.shape[0]):
    for j in range(U.shape[1]):
        for c in range(U.shape[3]):
            if U[i,j,0,c] < 0:
                U[i,j,:,c] *= -1
#Cx = scipy.ndimage.gaussian_filter(Cx_pre, sigma = (2,0,0), order = 0)
Cx = scipy.ndimage.median_filter(Cx_pre, size = 5, axes = 0)
n_channels = Cx.shape[2]

def C_interpolate(k, o, i):
    func = RectBivariateSpline(np.arange(Cx.shape[0]), np.arange(Cx.shape[1]), np.log10(Cx[:,:,i]), s = 0)
    return 10**func(k,o)/1000000000 #November 7th, 2024
    #return 10**func(k,o)/10000000

def C(kk, oo, c):
    size1 = kk.shape[0]
    size2 = kk.shape[1]
    Cx_broad = np.zeros(kk.shape)
    for i in range(size1):
        for j in range(size2):
            Cx_broad[i,j] = Cx[int(kk[i,j]), int(oo[i,j]),c]
    return Cx_broad

L = 100  # linear size of space in one dimension
T = 50 # size of time

#First frequencies are positive, I shouldn't use fftshift!!!
#freqs_k = scipy.fft.fftshift(scipy.fft.fftfreq(N, d=dz) * 2 * np.pi)
#freqs_omega = scipy.fft.fftshift(scipy.fft.fftfreq(M, d=dt) * 2 * np.pi)

k_low = 0 
o_low = 0 
freqs_k_small = np.linspace(50,60,10)
freqs_omega_small = np.linspace(50,60,10)

oo, kk = np.meshgrid(freqs_omega_small, freqs_k_small)


def create_information_eps(fixed_freq, freqs, fixed_type, return_sum = True):
    def information_eps(log_eps):
        infos = []
        for i in range(n_channels):
            eps_eigenchannel = 10**log_eps[i]
            if fixed_type == 'spatial':
                #ktilde = 1/Cx[int(fixed_freq),:, i]
                ktilde = 1/C_interpolate(int(fixed_freq), freqs, i)
            elif fixed_type == 'temporal':
                #ktilde = 1/Cx[:,int(fixed_freq), i]
                ktilde = 1/C_interpolate(freqs, int(fixed_freq), i)
            ktilde = pad_and_reflect(ktilde[:,0],ktilde.shape[0]*2 - 2)
            #numer = np.maximum(0, sigout**2 /(2 * (ktilde + 1e-32)) * (np.sqrt(1 + 4 * ktilde/(eps_eigenchannel * sigout**2)) - 1) - sigout**2) + sigout**2
            #denom = np.maximum(0, sigout**2 /(2 * (ktilde + 1)) * (np.sqrt(1 + 4 * ktilde/(eps_eigenchannel * sigout**2)) + 1) - sigout**2) + sigout**2

            numer = np.maximum(0, 1/(2 * (ktilde + 1e-32)) * (np.sqrt(1 + 4 * ktilde/(eps_eigenchannel * sigout**2)) - 1) - 1) + 1
            denom = np.maximum(0, 1/(2 * (ktilde + 1)) * (np.sqrt(1 + 4 * ktilde/(eps_eigenchannel * sigout**2)) + 1) - 1) + 1
            
            
            info = np.sum(np.log(numer) - np.log(denom))
            infos.append(info)
        if return_sum:
            return np.sum(infos)*-1
        else:
            return infos 
    return information_eps

def filter_k(fixed_freq, indices, eps, fixed_type, channel, k_lims=None, return_ktilde = False):
    def v_opt(indices):
        if fixed_type == 'spatial':
            inter = np.clip(C_interpolate(int(fixed_freq),indices, channel), 1e-32, np.inf).T
            ktilde = 1/inter
        elif fixed_type == 'temporal':
            inter = np.clip(C_interpolate(indices, int(fixed_freq), channel), 1e-32, np.inf)
            ktilde = 1/inter
        ktilde = pad_and_reflect(ktilde[:,0],ktilde.shape[0]*2 - 2)
        sqrt_piece = np.sqrt(1 + (4/eps) * ktilde)
        
        v2 = 0.5 * (sqrt_piece + 1) / (1 + ktilde) - 1
        v2 = np.sqrt(np.maximum(v2, 0) * sigout**2)
        if k_lims:
            unit_cell_k = soft_bandpass(k_lims[0], k_lims[1], indices)
            v2 *= unit_cell_k
        if not return_ktilde:
            return v2
        else:
            return ktilde
    return v_opt

def filter(C, nu, k_lims=None):
    def v_opt(k, o, i):
        CC = np.minimum(C_interpolate(k, o, i), 1e32)
        sqrt_piece = np.sqrt(CC**2 + (4/nu) * (1/sigout**2) * CC)
        v2 = 0.5 * (sqrt_piece + CC) / (1 + CC) - 1
        v2 = np.sqrt(np.maximum(v2, 0) * sigout**2)
        if k_lims:
            unit_cell_k = soft_bandpass(k_lims[0], k_lims[1], k)
            v2 *= unit_cell_k

        return v2 #- np.min(v2)
    return v_opt

def pad_and_reflect(filter, N, padval=0):
    """
    Zero-pad to length N//2 and reflect about origin to make length N. 
    """
    expanded = np.pad(filter, (0, N//2 + 1 - len(filter)), constant_values=(0, padval))
    return np.concatenate([expanded[::-1], expanded[1:]])

def extrap_and_reflect(filter, N, return_log=False):
    """
    Linearly extrapolate the log filter to better approximate tails and reflect 
    about origin to make length N.
    """
    extrap_fun = interp1d(range(0, len(filter)), np.maximum(-32, np.log(filter)), fill_value='extrapolate')
    expanded = extrap_fun(range(0, N//2 + 1))
    if not return_log:
        expanded = np.exp(expanded)
    return np.concatenate([expanded[::-1], expanded[1:]])

def filter_power(eps, fixed_freq, indices, klims, channel, fixed_type):
    vfun = filter_k(fixed_freq, indices, eps, fixed_type, channel, klims)
    v2 = vfun(indices)
    if fixed_type == 'spatial':
        ktilde = 1/C_interpolate(int(fixed_freq), indices, channel)
    elif fixed_type == 'temporal':
        ktilde = 1/C_interpolate(indices, int(fixed_freq), channel)
    
    ktilde = pad_and_reflect(ktilde[:,0],ktilde.shape[0]*2 - 2)

    dk = 2*np.pi/len(indices)

    freqs = indices*2*np.pi/len(indices)
    freqs = pad_and_reflect(freqs, len(freqs)*2 - 2)

    #return np.sum(v2**2 * np.abs(freqs) * (ktilde + 1) * dk)/(2 * np.pi)**2
    return np.sum(v2**2 * np.abs(freqs) * (1/ktilde + 1) * dk)/(2 * np.pi)
    #John: Needs to make sure I do the conversion from integral to sum correctly
    #Need to make srue dk is the correct value. dk = 2*pi/L
    #By convention, frequencies discrete are -pi to pi. My frequencies are from 0 to L. 

def excess_power(log_eps, fixed_freq, indices, klims, P, fixed_type):
    total_power = 0
    for i in range(n_channels):
        eig_power = filter_power(10**log_eps[i], fixed_freq, indices, klims, i, fixed_type)
        total_power += eig_power
    return P - total_power

def optimal_logeps(fixed_freq, indices, klims, P, fixed_type):
    power_constraint = {'type': 'ineq', 'fun': excess_power, 'args': (fixed_freq, indices, klims, P, fixed_type)}
    eps_fun = create_information_eps(fixed_freq, indices, fixed_type)
    res = opt.minimize(eps_fun, np.array([-10,-10,-10]), bounds=[(-16, np.inf)], constraints=[power_constraint])
    if res.success:
        #print("Excess power: ", excess_power(res.x, fixed_freq, indices, klims, P, fixed_type))
        return res.x
    else:
        print("Optimizer failed to converge!")
        return None
